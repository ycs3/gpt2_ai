import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    raise ImportError(
        "This script requires MLX. Install it first, for example: pip install mlx"
    ) from exc


TIMESTEPS = 500
BETA_START = 1e-4
BETA_END = 2e-2
IMG_SIZE = 48
NUM_FRAMES = 8
CHANNELS = 3
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
BASE_CHANNELS = 48
TIME_EMBED_DIM = 128
COND_DROP_PROB = 0.1
WEIGHTS_PATH = "image_to_video_diffusion_ucf101_mlx.safetensors"
DEFAULT_CLASSES = ["Basketball", "Biking", "HorseRiding", "Surfing", "Typing"]


def stable_softmax(x, axis=-1):
    shifted = x - mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(shifted)
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def count_params(tree):
    total = 0
    if isinstance(tree, dict):
        for value in tree.values():
            total += count_params(value)
        return total
    if isinstance(tree, list):
        for value in tree:
            total += count_params(value)
        return total
    if hasattr(tree, "shape"):
        return int(np.prod(tree.shape))
    return 0


def parse_classes_arg(classes_arg):
    if not classes_arg:
        return list(DEFAULT_CLASSES)
    return [item.strip() for item in classes_arg.split(",") if item.strip()]


def load_split_paths(split_file, videos_dir):
    video_root = Path(videos_dir)
    samples = []
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path = line.split()[0]
            samples.append(video_root / rel_path)
    return samples


def discover_video_paths(videos_dir):
    video_root = Path(videos_dir)
    return sorted(path for path in video_root.rglob("*.avi") if path.is_file())


def build_class_mapping(video_paths, selected_classes=None):
    class_names = sorted({path.parent.name for path in video_paths})
    if selected_classes:
        class_names = [name for name in selected_classes if name in class_names]
    if not class_names:
        raise ValueError("No matching classes were found in the dataset directory.")
    return {name: idx for idx, name in enumerate(class_names)}


def split_samples(video_paths, train_ratio=0.8, seed=0):
    grouped = {}
    for path in video_paths:
        grouped.setdefault(path.parent.name, []).append(path)

    rng = np.random.default_rng(seed)
    train_paths = []
    test_paths = []
    for paths in grouped.values():
        paths = list(paths)
        rng.shuffle(paths)
        cutoff = max(1, int(len(paths) * train_ratio))
        cutoff = min(cutoff, len(paths) - 1) if len(paths) > 1 else len(paths)
        train_paths.extend(paths[:cutoff])
        test_paths.extend(paths[cutoff:] if len(paths) > 1 else paths[:1])
    return sorted(train_paths), sorted(test_paths)


def filter_and_limit_paths(video_paths, class_to_idx, max_videos_per_class=None):
    counts = {name: 0 for name in class_to_idx}
    filtered = []
    for path in video_paths:
        class_name = path.parent.name
        if class_name not in class_to_idx:
            continue
        if max_videos_per_class is not None and counts[class_name] >= max_videos_per_class:
            continue
        filtered.append(path)
        counts[class_name] += 1
    return filtered


def gather_dataset(
    videos_dir,
    selected_classes,
    train_split_file=None,
    test_split_file=None,
    max_train_videos_per_class=None,
    max_test_videos_per_class=None,
    seed=0,
):
    if train_split_file and test_split_file:
        train_candidates = load_split_paths(train_split_file, videos_dir)
        test_candidates = load_split_paths(test_split_file, videos_dir)
        class_to_idx = build_class_mapping(train_candidates + test_candidates, selected_classes)
        train_paths = filter_and_limit_paths(
            train_candidates,
            class_to_idx,
            max_videos_per_class=max_train_videos_per_class,
        )
        test_paths = filter_and_limit_paths(
            test_candidates,
            class_to_idx,
            max_videos_per_class=max_test_videos_per_class,
        )
    else:
        discovered = discover_video_paths(videos_dir)
        class_to_idx = build_class_mapping(discovered, selected_classes)
        discovered = filter_and_limit_paths(discovered, class_to_idx)
        train_candidates, test_candidates = split_samples(discovered, seed=seed)
        train_paths = filter_and_limit_paths(
            train_candidates,
            class_to_idx,
            max_videos_per_class=max_train_videos_per_class,
        )
        test_paths = filter_and_limit_paths(
            test_candidates,
            class_to_idx,
            max_videos_per_class=max_test_videos_per_class,
        )

    if not train_paths:
        raise ValueError("Training split is empty. Check the dataset path, class list, or split files.")

    class_names = sorted(class_to_idx, key=lambda name: class_to_idx[name])
    train_labels = np.array([class_to_idx[path.parent.name] for path in train_paths], dtype=np.int32)
    test_labels = np.array([class_to_idx[path.parent.name] for path in test_paths], dtype=np.int32)
    return train_paths, train_labels, test_paths, test_labels, class_names


def probe_duration(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def decode_video_ffmpeg(video_path, num_frames, image_size):
    duration = probe_duration(video_path)
    fps = max(num_frames / duration, 1.0) if duration else num_frames
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.4f},scale={image_size}:{image_size}:flags=bicubic",
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed for {video_path}: {error}")

    frame_size = image_size * image_size * 3
    buffer = result.stdout
    frame_count = len(buffer) // frame_size
    if frame_count == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    frames = np.frombuffer(buffer[: frame_count * frame_size], dtype=np.uint8)
    frames = frames.reshape(frame_count, image_size, image_size, 3).astype(np.float32)

    if frame_count < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frame_count, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    elif frame_count > num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames).round().astype(np.int32)
        frames = frames[indices]

    return frames / 127.5 - 1.0


def load_video_dataset(video_paths, labels, num_frames, image_size):
    clips = []
    kept_labels = []
    kept_paths = []
    for path, label in zip(video_paths, labels):
        try:
            clip = decode_video_ffmpeg(path, num_frames=num_frames, image_size=image_size)
        except RuntimeError as exc:
            print(f"warning: skipping {path} ({exc})")
            continue
        clips.append(clip)
        kept_labels.append(label)
        kept_paths.append(path)

    if not clips:
        raise ValueError("No videos could be decoded. Verify ffmpeg and the dataset files.")

    return np.stack(clips), np.array(kept_labels, dtype=np.int32), kept_paths


def load_image(image_path, image_size):
    pixels = plt.imread(image_path)
    if pixels.ndim == 2:
        pixels = np.stack([pixels, pixels, pixels], axis=-1)
    if pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    pixels = pixels.astype(np.float32)
    if pixels.max() > 1.0:
        pixels = pixels / 255.0

    if pixels.shape[0] != image_size or pixels.shape[1] != image_size:
        fig = plt.figure(figsize=(1, 1), dpi=image_size)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.imshow(np.clip(pixels, 0.0, 1.0))
        axis.axis("off")
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rendered = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        plt.close(fig)
        pixels = rendered.astype(np.float32) / 255.0

    return pixels * 2.0 - 1.0


def batch_iterator(clips, labels, batch_size, shuffle=True, drop_last=True):
    indices = np.arange(len(clips))
    if shuffle:
        indices = np.random.permutation(indices)

    limit = len(indices)
    if drop_last:
        limit = (limit // batch_size) * batch_size

    for start in range(0, limit, batch_size):
        batch_ids = indices[start : start + batch_size]
        if len(batch_ids) < batch_size and drop_last:
            continue
        yield mx.array(clips[batch_ids]), mx.array(labels[batch_ids])


@dataclass
class DiffusionSchedule:
    betas: mx.array
    alphas: mx.array
    alpha_cumprod: mx.array
    alpha_cumprod_prev: mx.array
    sqrt_alpha_cumprod: mx.array
    sqrt_one_minus_alpha_cumprod: mx.array


def make_schedule():
    betas = mx.linspace(BETA_START, BETA_END, TIMESTEPS)
    alphas = 1.0 - betas
    alpha_cumprod = mx.cumprod(alphas, axis=0)
    alpha_cumprod_prev = mx.concatenate([mx.array([1.0]), alpha_cumprod[:-1]], axis=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_cumprod=alpha_cumprod,
        alpha_cumprod_prev=alpha_cumprod_prev,
        sqrt_alpha_cumprod=mx.sqrt(alpha_cumprod),
        sqrt_one_minus_alpha_cumprod=mx.sqrt(1.0 - alpha_cumprod),
    )


def forward_diffusion(x0, timesteps, schedule, noise=None):
    if noise is None:
        noise = mx.random.normal(shape=x0.shape)
    scale_clean = schedule.sqrt_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1, 1)
    scale_noise = schedule.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1, 1)
    return scale_clean * x0 + scale_noise * noise, noise


def make_sampling_timesteps(sample_steps):
    if sample_steps is None or sample_steps >= TIMESTEPS:
        return list(range(TIMESTEPS - 1, -1, -1))

    if sample_steps <= 1:
        raise ValueError("--sample-steps must be at least 2")

    steps = np.linspace(0, TIMESTEPS - 1, sample_steps)
    steps = np.unique(np.round(steps).astype(np.int32))
    if steps[-1] != TIMESTEPS - 1:
        steps = np.append(steps, TIMESTEPS - 1)
    return list(steps[::-1])


def apply_per_frame(module, x):
    batch_size, num_frames, height, width, channels = x.shape
    y = module(x.reshape(batch_size * num_frames, height, width, channels))
    out_h, out_w, out_c = y.shape[1:]
    return y.reshape(batch_size, num_frames, out_h, out_w, out_c)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, timesteps):
        half_dim = self.dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half_dim, dtype=mx.float32) / max(half_dim, 1)
        )
        args = timesteps.astype(mx.float32).reshape(-1, 1) * freqs.reshape(1, -1)
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        return emb


class VideoResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels, pytorch_compatible=True)
        self.norm2 = nn.GroupNorm(8, out_channels, pytorch_compatible=True)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x, emb):
        h = apply_per_frame(self.conv1, x)
        h = apply_per_frame(lambda y: self.norm1(nn.silu(y)), h)
        h = h + self.emb_proj(nn.silu(emb)).reshape(emb.shape[0], 1, 1, 1, -1)
        h = apply_per_frame(self.conv2, h)
        h = apply_per_frame(lambda y: self.norm2(nn.silu(y)), h)
        residual = apply_per_frame(self.skip, x) if self.skip is not None else x
        return h + residual


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)
        self.channels = channels

    def __call__(self, x):
        batch_size, num_frames, height, width, channels = x.shape
        h = self.norm(x.reshape(batch_size * num_frames, height * width, channels))
        qkv = self.qkv(h).reshape(batch_size * num_frames, height * width, 3, channels)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        attn = stable_softmax((q @ k.transpose(0, 2, 1)) / math.sqrt(channels), axis=-1)
        out = self.out(attn @ v).reshape(batch_size, num_frames, height, width, channels)
        return x + out


class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)

    def __call__(self, x):
        batch_size, num_frames, height, width, channels = x.shape
        h = x.transpose(0, 2, 3, 1, 4).reshape(batch_size * height * width, num_frames, channels)
        h = self.norm(h)
        qkv = self.qkv(h).reshape(
            batch_size * height * width,
            num_frames,
            3,
            self.num_heads,
            self.head_dim,
        )
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = stable_softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_size * height * width, num_frames, channels)
        out = self.out(out).reshape(batch_size, height, width, num_frames, channels).transpose(0, 3, 1, 2, 4)
        return x + out


class ImageConditionedVideoUNet(nn.Module):
    def __init__(self, in_channels=CHANNELS, base_channels=BASE_CHANNELS, time_dim=TIME_EMBED_DIM):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_gate = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.conv_in = nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1)
        self.down1_res = VideoResBlock(base_channels, base_channels, time_dim)
        self.down1_pool = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        self.down2_res = VideoResBlock(base_channels, base_channels * 2, time_dim)
        self.down2_pool = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.mid_res1 = VideoResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid_spatial_attn = SpatialAttention(base_channels * 2)
        self.mid_temporal_attn = TemporalAttention(base_channels * 2)
        self.mid_res2 = VideoResBlock(base_channels * 2, base_channels * 2, time_dim)

        self.up2_upsample = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.up2_res = VideoResBlock(base_channels * 4, base_channels, time_dim)

        self.up1_upsample = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.up1_res = VideoResBlock(base_channels * 2, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(8, base_channels, pytorch_compatible=True)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, noisy_clips, reference_images, timesteps, cond_mask=None):
        time_emb = self.time_embed(timesteps)
        if cond_mask is None:
            cond_mask = mx.ones((noisy_clips.shape[0], 1), dtype=time_emb.dtype)
        else:
            cond_mask = cond_mask.astype(time_emb.dtype).reshape(-1, 1)
        emb = self.cond_gate(time_emb * cond_mask)

        ref_frames = mx.broadcast_to(
            reference_images.reshape(reference_images.shape[0], 1, reference_images.shape[1], reference_images.shape[2], reference_images.shape[3]),
            noisy_clips.shape,
        )
        x = mx.concatenate([noisy_clips, ref_frames * cond_mask.reshape(-1, 1, 1, 1, 1)], axis=-1)

        h = apply_per_frame(self.conv_in, x)
        h1 = self.down1_res(h, emb)
        h = apply_per_frame(self.down1_pool, h1)

        h2 = self.down2_res(h, emb)
        h = apply_per_frame(self.down2_pool, h2)

        h = self.mid_res1(h, emb)
        h = self.mid_spatial_attn(h)
        h = self.mid_temporal_attn(h)
        h = self.mid_res2(h, emb)

        h = apply_per_frame(self.up2_upsample, h)
        h = mx.concatenate([h, h2], axis=-1)
        h = self.up2_res(h, emb)

        h = apply_per_frame(self.up1_upsample, h)
        h = mx.concatenate([h, h1], axis=-1)
        h = self.up1_res(h, emb)

        h = apply_per_frame(lambda y: self.out_norm(nn.silu(y)), h)
        return apply_per_frame(self.out_conv, h)


def loss_fn(model, clips, schedule):
    batch_size = clips.shape[0]
    timesteps = mx.random.randint(0, TIMESTEPS, shape=(batch_size,), dtype=mx.int32)
    keep_mask = mx.random.uniform(shape=(batch_size,)) > COND_DROP_PROB
    reference_images = clips[:, 0]
    noisy_clips, true_noise = forward_diffusion(clips, timesteps, schedule)
    noisy_clips = mx.concatenate(
        [reference_images.reshape(batch_size, 1, clips.shape[2], clips.shape[3], clips.shape[4]), noisy_clips[:, 1:]],
        axis=1,
    )
    predicted_noise = model(noisy_clips, reference_images, timesteps, cond_mask=keep_mask)
    zero_noise = mx.zeros_like(true_noise[:, :1])
    target_noise = mx.concatenate([zero_noise, true_noise[:, 1:]], axis=1)
    return nn.losses.mse_loss(predicted_noise, target_noise)


def show_video_grid(samples, reference_images=None, title="Image-to-video samples", save_path=None, show=True):
    clips = np.clip((np.array(samples) * 0.5) + 0.5, 0.0, 1.0)
    count = len(clips)
    num_frames = clips.shape[1]
    extra_col = 1 if reference_images is not None else 0
    fig, axes = plt.subplots(count, num_frames + extra_col, figsize=((num_frames + extra_col) * 1.4, count * 1.8))
    axes = np.array(axes).reshape(count, num_frames + extra_col)

    refs = None if reference_images is None else np.clip((np.array(reference_images) * 0.5) + 0.5, 0.0, 1.0)
    for row in range(count):
        col_offset = 0
        if refs is not None:
            axes[row, 0].imshow(refs[row])
            axes[row, 0].set_title("ref", fontsize=8)
            axes[row, 0].axis("off")
            col_offset = 1
        for col in range(num_frames):
            axes[row, col + col_offset].imshow(clips[row, col])
            axes[row, col + col_offset].axis("off")
            if col == 0:
                axes[row, col + col_offset].set_title("frame 0", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def show_forward_process(clips, schedule, save_path="forward_image_to_video_diffusion_mlx.png", show=True):
    clip = mx.array(clips[:1])
    steps = [0, 25, 100, 250, 499]
    fig, axes = plt.subplots(len(steps), clip.shape[1], figsize=(clip.shape[1] * 1.4, len(steps) * 1.6))
    axes = np.array(axes).reshape(len(steps), clip.shape[1])

    for row, timestep in enumerate(steps):
        noisy, _ = forward_diffusion(clip, mx.array([timestep], dtype=mx.int32), schedule)
        pixels = np.clip(np.array(noisy[0]) * 0.5 + 0.5, 0.0, 1.0)
        for col in range(clip.shape[1]):
            axes[row, col].imshow(pixels[col])
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(f"t={timestep}", fontsize=8)

    fig.suptitle("Forward diffusion on training clip", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_training_data(args):
    selected_classes = parse_classes_arg(args.classes)
    train_paths, train_labels, test_paths, test_labels, class_names = gather_dataset(
        videos_dir=args.videos_dir,
        selected_classes=selected_classes,
        train_split_file=args.train_split_file,
        test_split_file=args.test_split_file,
        max_train_videos_per_class=args.max_train_videos_per_class,
        max_test_videos_per_class=args.max_test_videos_per_class,
        seed=args.seed,
    )
    print(f"classes: {', '.join(class_names)}")
    print(f"loading {len(train_paths)} training videos")
    train_clips, train_labels, train_paths = load_video_dataset(
        train_paths, train_labels, args.num_frames, args.image_size
    )
    test_clips = None
    if test_paths:
        print(f"loading {len(test_paths)} held-out videos")
        test_clips, _, _ = load_video_dataset(test_paths, test_labels, args.num_frames, args.image_size)
    return train_clips, test_clips, train_paths


def load_dataset_metadata(args):
    selected_classes = parse_classes_arg(args.classes)
    train_paths, train_labels, test_paths, test_labels, class_names = gather_dataset(
        videos_dir=args.videos_dir,
        selected_classes=selected_classes,
        train_split_file=args.train_split_file,
        test_split_file=args.test_split_file,
        max_train_videos_per_class=args.max_train_videos_per_class,
        max_test_videos_per_class=args.max_test_videos_per_class,
        seed=args.seed,
    )
    return train_paths, train_labels, test_paths, test_labels, class_names


def sample(
    model,
    schedule,
    reference_images,
    sample_steps=None,
    ddim_eta=0.0,
    image_size=IMG_SIZE,
    num_frames=NUM_FRAMES,
):
    model.eval()
    if reference_images.ndim == 3:
        reference_images = reference_images[None, ...]

    sample_count = int(reference_images.shape[0])
    x = mx.random.normal(shape=(sample_count, num_frames, image_size, image_size, CHANNELS))
    x = mx.concatenate([reference_images.reshape(sample_count, 1, image_size, image_size, CHANNELS), x[:, 1:]], axis=1)
    sampling_timesteps = make_sampling_timesteps(sample_steps)

    for step_idx, timestep in enumerate(sampling_timesteps):
        next_timestep = sampling_timesteps[step_idx + 1] if step_idx + 1 < len(sampling_timesteps) else -1
        t = mx.full((sample_count,), timestep, dtype=mx.int32)

        pred_noise = model(x, reference_images, t)
        zero_noise = mx.zeros_like(pred_noise[:, :1])
        pred_noise = mx.concatenate([zero_noise, pred_noise[:, 1:]], axis=1)

        alpha_bar_t = schedule.alpha_cumprod[timestep]
        alpha_bar_next = mx.array(1.0) if next_timestep < 0 else schedule.alpha_cumprod[next_timestep]
        pred_x0 = (x - mx.sqrt(1.0 - alpha_bar_t) * pred_noise) / mx.sqrt(alpha_bar_t)
        pred_x0 = mx.concatenate([reference_images.reshape(sample_count, 1, image_size, image_size, CHANNELS), pred_x0[:, 1:]], axis=1)

        if next_timestep < 0:
            x = pred_x0
            continue

        sigma = ddim_eta * mx.sqrt(
            ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t))
            * (1.0 - alpha_bar_t / alpha_bar_next)
        )
        direction = mx.sqrt(mx.maximum(1.0 - alpha_bar_next - sigma**2, 0.0)) * pred_noise
        x = mx.sqrt(alpha_bar_next) * pred_x0 + direction
        if float(np.array(sigma)) > 0.0:
            x = x + sigma * mx.random.normal(shape=x.shape)
        x = mx.concatenate([reference_images.reshape(sample_count, 1, image_size, image_size, CHANNELS), x[:, 1:]], axis=1)

    model.train()
    return mx.clip(x, -1.0, 1.0)


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_clips, test_clips, _ = load_training_data(args)
    schedule = make_schedule()
    model = ImageConditionedVideoUNet(
        in_channels=CHANNELS,
        base_channels=args.base_channels,
        time_dim=args.time_embed_dim,
    )
    if args.resume_from:
        model.load_weights(args.resume_from)
        print(f"loaded checkpoint from {args.resume_from}")
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(f"model parameters: {count_params(model.parameters()) / 1e6:.2f}M")
    print(f"training clips: {len(train_clips)}")
    if test_clips is not None:
        print(f"held-out clips: {len(test_clips)}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        seen_steps = 0
        for batch_clips, _ in batch_iterator(
            train_clips,
            np.zeros((len(train_clips),), dtype=np.int32),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        ):
            loss, grads = loss_and_grad_fn(model, batch_clips, schedule)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total_loss += float(np.array(loss))
            seen_steps += 1

        avg_loss = total_loss / max(seen_steps, 1)
        print(f"epoch {epoch:>2d}/{args.epochs} avg_loss={avg_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == 1:
            preview_source = train_clips[: args.preview_count, 0]
            preview = sample(
                model,
                schedule,
                reference_images=mx.array(preview_source),
                sample_steps=args.sample_steps,
                ddim_eta=args.ddim_eta,
                image_size=args.image_size,
                num_frames=args.num_frames,
            )
            show_video_grid(
                preview,
                reference_images=preview_source,
                title=f"Epoch {epoch}",
                save_path=f"samples_image_to_video_epoch{epoch:02d}.png",
                show=not args.no_show,
            )

    model.save_weights(args.weights_path)
    print(f"saved model weights to {args.weights_path}")
    return model, schedule, train_clips


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or sample a small image-to-video diffusion model in MLX using UCF101 clips."
    )
    parser.add_argument("--mode", choices=["train", "sample", "forward"], default="train")
    parser.add_argument("--videos-dir", type=str, required=True)
    parser.add_argument("--train-split-file", type=str, default=None)
    parser.add_argument("--test-split-file", type=str, default=None)
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--max-train-videos-per-class", type=int, default=40)
    parser.add_argument("--max-test-videos-per-class", type=int, default=8)
    parser.add_argument("--weights-path", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--time-embed-dim", type=int, default=TIME_EMBED_DIM)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=IMG_SIZE)
    parser.add_argument("--input-image", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        model, schedule, train_clips = train(args)
        preview_refs = mx.array(train_clips[: args.preview_count, 0])
        samples = sample(
            model,
            schedule,
            reference_images=preview_refs,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
            image_size=args.image_size,
            num_frames=args.num_frames,
        )
        show_video_grid(
            samples,
            reference_images=preview_refs,
            title="Post-training image-to-video samples",
            save_path="samples_image_to_video_mlx.png",
            show=not args.no_show,
        )
        if len(train_clips) > 0:
            show_forward_process(train_clips, schedule, show=not args.no_show)
        return

    train_paths, _, _, _, _ = load_dataset_metadata(args)
    schedule = make_schedule()
    model = ImageConditionedVideoUNet(
        in_channels=CHANNELS,
        base_channels=args.base_channels,
        time_dim=args.time_embed_dim,
    )
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())

    if args.mode == "forward":
        train_clips, _, _ = load_video_dataset(
            train_paths[:1],
            np.zeros((min(1, len(train_paths)),), dtype=np.int32),
            args.num_frames,
            args.image_size,
        )
        show_forward_process(train_clips, schedule, show=not args.no_show)
        return

    if args.input_image:
        reference_image = load_image(args.input_image, args.image_size)
    else:
        train_clips, _, _ = load_video_dataset(
            train_paths[:1],
            np.zeros((min(1, len(train_paths)),), dtype=np.int32),
            args.num_frames,
            args.image_size,
        )
        reference_image = train_clips[0, 0]

    reference_images = mx.array(reference_image.reshape(1, args.image_size, args.image_size, CHANNELS))
    samples = sample(
        model,
        schedule,
        reference_images=reference_images,
        sample_steps=args.sample_steps,
        ddim_eta=args.ddim_eta,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )
    show_video_grid(
        samples,
        reference_images=reference_images,
        title="Image-conditioned video sample",
        save_path="samples_image_to_video_mlx.png",
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
