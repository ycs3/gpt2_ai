import argparse
import math
import subprocess
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


IMG_SIZE = 48
NUM_FRAMES = 8
CHANNELS = 3
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
BASE_CHANNELS = 48
LATENT_CHANNELS = 32
KL_WEIGHT = 1e-4
WEIGHTS_PATH = "video_vae_ucf101_mlx.safetensors"
DEFAULT_CLASSES = ["Basketball", "Biking", "HorseRiding", "Surfing", "Typing"]


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


def stable_softmax(x, axis=-1):
    shifted = x - mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(shifted)
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


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
    for path, label in zip(video_paths, labels):
        try:
            clip = decode_video_ffmpeg(path, num_frames=num_frames, image_size=image_size)
        except RuntimeError as exc:
            print(f"warning: skipping {path} ({exc})")
            continue
        clips.append(clip)
        kept_labels.append(label)

    if not clips:
        raise ValueError("No videos could be decoded. Verify ffmpeg and the dataset files.")

    return np.stack(clips), np.array(kept_labels, dtype=np.int32)


def batch_iterator(clips, labels, batch_size, shuffle=True, drop_last=False):
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


def apply_per_frame(module, x):
    batch_size, num_frames, height, width, channels = x.shape
    y = module(x.reshape(batch_size * num_frames, height, width, channels))
    out_h, out_w, out_c = y.shape[1:]
    return y.reshape(batch_size, num_frames, out_h, out_w, out_c)


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


class VideoEncoder(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS, latent_channels=LATENT_CHANNELS):
        super().__init__()
        self.conv_in = nn.Conv2d(CHANNELS, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.temporal_attn = TemporalAttention(base_channels * 2)
        self.mu_proj = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=1)
        self.logvar_proj = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=1)

    def __call__(self, x):
        h = apply_per_frame(self.conv_in, x)
        h = nn.silu(h)
        h = apply_per_frame(self.down1, h)
        h = nn.silu(h)
        h = apply_per_frame(self.down2, h)
        h = nn.silu(h)
        h = self.temporal_attn(h)
        mu = apply_per_frame(self.mu_proj, h)
        logvar = apply_per_frame(self.logvar_proj, h)
        return mu, mx.clip(logvar, -10.0, 10.0)


class VideoDecoder(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS, latent_channels=LATENT_CHANNELS):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 2, kernel_size=3, padding=1)
        self.temporal_attn = TemporalAttention(base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(base_channels, CHANNELS, kernel_size=3, padding=1)

    def __call__(self, z):
        h = apply_per_frame(self.conv_in, z)
        h = nn.silu(h)
        h = self.temporal_attn(h)
        h = apply_per_frame(self.up2, h)
        h = nn.silu(h)
        h = apply_per_frame(self.up1, h)
        h = nn.silu(h)
        return mx.tanh(apply_per_frame(self.conv_out, h))


class VideoVAE(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS, latent_channels=LATENT_CHANNELS):
        super().__init__()
        self.encoder = VideoEncoder(base_channels=base_channels, latent_channels=latent_channels)
        self.decoder = VideoDecoder(base_channels=base_channels, latent_channels=latent_channels)
        self.latent_channels = latent_channels

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(shape=mu.shape)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def kl_divergence(mu, logvar):
    return -0.5 * mx.mean(1.0 + logvar - mu * mu - mx.exp(logvar))


def reconstruction_loss(recon, target):
    return nn.losses.mse_loss(recon, target)


def loss_fn(model, clips, kl_weight):
    recon, mu, logvar = model(clips)
    recon_loss = reconstruction_loss(recon, clips)
    kl_loss = kl_divergence(mu, logvar)
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def evaluate(model, clips, labels, batch_size, kl_weight):
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_seen = 0
    for batch_clips, _ in batch_iterator(clips, labels, batch_size=batch_size, shuffle=False, drop_last=False):
        recon, mu, logvar = model(batch_clips)
        recon_loss = reconstruction_loss(recon, batch_clips)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + kl_weight * kl_loss
        mx.eval(loss, recon_loss, kl_loss)
        batch_size_actual = int(batch_clips.shape[0])
        total_loss += float(np.array(loss)) * batch_size_actual
        total_recon += float(np.array(recon_loss)) * batch_size_actual
        total_kl += float(np.array(kl_loss)) * batch_size_actual
        total_seen += batch_size_actual
    denom = max(total_seen, 1)
    return total_loss / denom, total_recon / denom, total_kl / denom


def show_reconstructions(originals, reconstructions, save_path=None, show=True, title="Video VAE reconstructions"):
    originals = np.clip((np.array(originals) * 0.5) + 0.5, 0.0, 1.0)
    reconstructions = np.clip((np.array(reconstructions) * 0.5) + 0.5, 0.0, 1.0)
    count = len(originals)
    num_frames = originals.shape[1]
    fig, axes = plt.subplots(count * 2, num_frames, figsize=(num_frames * 1.4, count * 2.2))
    axes = np.array(axes).reshape(count * 2, num_frames)

    for row in range(count):
        for col in range(num_frames):
            axes[row * 2, col].imshow(originals[row, col])
            axes[row * 2, col].axis("off")
            axes[row * 2 + 1, col].imshow(reconstructions[row, col])
            axes[row * 2 + 1, col].axis("off")
            if col == 0:
                axes[row * 2, col].set_title("orig", fontsize=8)
                axes[row * 2 + 1, col].set_title("recon", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def show_samples(samples, save_path=None, show=True, title="Video VAE random samples"):
    clips = np.clip((np.array(samples) * 0.5) + 0.5, 0.0, 1.0)
    count = len(clips)
    num_frames = clips.shape[1]
    fig, axes = plt.subplots(count, num_frames, figsize=(num_frames * 1.4, count * 1.8))
    axes = np.array(axes).reshape(count, num_frames)

    for row in range(count):
        for col in range(num_frames):
            axes[row, col].imshow(clips[row, col])
            axes[row, col].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_datasets(args):
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
    train_clips, train_labels = load_video_dataset(train_paths, train_labels, args.num_frames, args.image_size)
    test_clips = None
    test_labels_loaded = None
    if test_paths:
        print(f"loading {len(test_paths)} held-out videos")
        test_clips, test_labels_loaded = load_video_dataset(test_paths, test_labels, args.num_frames, args.image_size)
    return train_clips, train_labels, test_clips, test_labels_loaded


def sample_from_prior(model, num_samples, num_frames, image_size):
    latent_h = image_size // 4
    latent_w = image_size // 4
    z = mx.random.normal(shape=(num_samples, num_frames, latent_h, latent_w, model.latent_channels))
    return model.decode(z)


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_clips, train_labels, test_clips, test_labels = load_datasets(args)
    model = VideoVAE(base_channels=args.base_channels, latent_channels=args.latent_channels)
    if args.resume_from:
        model.load_weights(args.resume_from)
        print(f"loaded checkpoint from {args.resume_from}")
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def total_loss_only(model_ref, batch_clips):
        total_loss, _, _ = loss_fn(model_ref, batch_clips, args.kl_weight)
        return total_loss

    loss_and_grad_fn = nn.value_and_grad(model, total_loss_only)

    print(f"model parameters: {count_params(model.parameters()) / 1e6:.2f}M")
    print(f"training clips: {len(train_clips)}")
    if test_clips is not None:
        print(f"held-out clips: {len(test_clips)}")

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        seen_steps = 0
        for batch_clips, _ in batch_iterator(
            train_clips,
            train_labels,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        ):
            loss, grads = loss_and_grad_fn(model, batch_clips)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            running_loss += float(np.array(loss))
            seen_steps += 1

        avg_train_loss = running_loss / max(seen_steps, 1)
        print(f"epoch {epoch:>2d}/{args.epochs} train_loss={avg_train_loss:.4f}")

        if test_clips is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            val_loss, val_recon, val_kl = evaluate(
                model,
                test_clips,
                test_labels,
                batch_size=args.eval_batch_size,
                kl_weight=args.kl_weight,
            )
            print(
                f"           val_loss={val_loss:.4f} "
                f"val_recon={val_recon:.4f} "
                f"val_kl={val_kl:.4f}"
            )

        if epoch % args.preview_every == 0 or epoch == 1:
            preview = mx.array(train_clips[: args.preview_count])
            recon, _, _ = model(preview)
            mx.eval(recon)
            show_reconstructions(
                preview,
                recon,
                save_path=f"video_vae_recon_epoch{epoch:02d}.png",
                show=not args.no_show,
                title=f"Epoch {epoch}",
            )

    model.save_weights(args.weights_path)
    print(f"saved model weights to {args.weights_path}")
    return model, train_clips


def build_parser():
    parser = argparse.ArgumentParser(description="Train or use a small UCF101 video VAE in MLX.")
    parser.add_argument("--mode", choices=["train", "reconstruct", "sample"], default="train")
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
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--latent-channels", type=int, default=LATENT_CHANNELS)
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=IMG_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        model, train_clips = train(args)
        preview = mx.array(train_clips[: args.preview_count])
        recon, _, _ = model(preview)
        mx.eval(recon)
        show_reconstructions(
            preview,
            recon,
            save_path="video_vae_reconstructions.png",
            show=not args.no_show,
            title="Post-training reconstructions",
        )
        prior_samples = sample_from_prior(
            model,
            num_samples=args.num_samples,
            num_frames=args.num_frames,
            image_size=args.image_size,
        )
        mx.eval(prior_samples)
        show_samples(
            prior_samples,
            save_path="video_vae_samples.png",
            show=not args.no_show,
        )
        return

    train_clips, train_labels, test_clips, test_labels = load_datasets(args)
    model = VideoVAE(base_channels=args.base_channels, latent_channels=args.latent_channels)
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())

    if args.mode == "reconstruct":
        source_clips = test_clips if test_clips is not None else train_clips
        preview = mx.array(source_clips[: args.preview_count])
        recon, _, _ = model(preview)
        mx.eval(recon)
        show_reconstructions(
            preview,
            recon,
            save_path="video_vae_reconstructions.png",
            show=not args.no_show,
        )
        return

    samples = sample_from_prior(
        model,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        image_size=args.image_size,
    )
    mx.eval(samples)
    show_samples(
        samples,
        save_path="video_vae_samples.png",
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
