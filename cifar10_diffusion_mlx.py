import argparse
import math
import pickle
import tarfile
import urllib.request
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


TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 2e-2
IMG_SIZE = 32
CHANNELS = 3
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
BASE_CHANNELS = 64
TIME_EMBED_DIM = 128
COND_DROP_PROB = 0.1
GUIDANCE_SCALE = 3.0
WEIGHTS_PATH = "diffusion_cifar10_labels_mlx.safetensors"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


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
    scale_clean = schedule.sqrt_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)
    scale_noise = schedule.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)
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


class ResBlock(nn.Module):
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
        h = self.conv1(x)
        h = self.norm1(nn.silu(h))
        h = h + self.emb_proj(nn.silu(emb)).reshape(emb.shape[0], 1, 1, -1)
        h = self.conv2(h)
        h = self.norm2(nn.silu(h))
        residual = self.skip(x) if self.skip is not None else x
        return h + residual


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels, pytorch_compatible=True)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(batch_size, height * width, 3, channels)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        attn = stable_softmax((q @ k.transpose(0, 2, 1)) / math.sqrt(channels), axis=-1)
        out = (attn @ v).reshape(batch_size, height, width, channels)
        return x + self.out(out)


class SmallUNet(nn.Module):
    def __init__(
        self,
        in_channels=CHANNELS,
        base_channels=BASE_CHANNELS,
        time_dim=TIME_EMBED_DIM,
        num_classes=len(CLASS_NAMES),
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes, time_dim)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1_res = ResBlock(base_channels, base_channels, time_dim)
        self.down1_pool = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        self.down2_res = ResBlock(base_channels, base_channels * 2, time_dim)
        self.down2_pool = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.mid_res1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid_attn = SelfAttention(base_channels * 2)
        self.mid_res2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)

        self.up2_upsample = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.up2_res = ResBlock(base_channels * 4, base_channels, time_dim)

        self.up1_upsample = nn.ConvTranspose2d(
            base_channels,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.up1_res = ResBlock(base_channels * 2, base_channels, time_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels, pytorch_compatible=True),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def __call__(self, x, timesteps, labels=None, cond_mask=None):
        time_emb = self.time_embed(timesteps)
        if labels is None:
            class_emb = mx.zeros_like(time_emb)
        else:
            class_emb = self.class_embed(labels)

        if cond_mask is None:
            cond_mask = mx.ones((x.shape[0], 1), dtype=time_emb.dtype)
        else:
            cond_mask = cond_mask.astype(time_emb.dtype).reshape(-1, 1)

        emb = self.cond_mlp(time_emb + class_emb * cond_mask)

        h = self.conv_in(x)
        h1 = self.down1_res(h, emb)
        h = self.down1_pool(h1)

        h2 = self.down2_res(h, emb)
        h = self.down2_pool(h2)

        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        h = self.up2_upsample(h)
        h = mx.concatenate([h, h2], axis=-1)
        h = self.up2_res(h, emb)

        h = self.up1_upsample(h)
        h = mx.concatenate([h, h1], axis=-1)
        h = self.up1_res(h, emb)

        return self.conv_out(h)


def ensure_cifar10(root):
    root = Path(root)
    archive_path = root / "cifar-10-python.tar.gz"
    extracted_dir = root / "cifar-10-batches-py"

    if extracted_dir.exists():
        return extracted_dir

    root.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"downloading CIFAR-10 to {archive_path}")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)

    print(f"extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)
    return extracted_dir


def load_cifar10_train(root="./data", download=True):
    root_path = Path(root)
    extracted_dir = root_path / "cifar-10-batches-py"
    if not extracted_dir.exists():
        if not download:
            raise FileNotFoundError(f"missing CIFAR-10 data at {extracted_dir}")
        extracted_dir = ensure_cifar10(root_path)

    image_batches = []
    label_batches = []
    for batch_idx in range(1, 6):
        batch_path = extracted_dir / f"data_batch_{batch_idx}"
        with open(batch_path, "rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images = batch[b"data"].reshape(-1, CHANNELS, IMG_SIZE, IMG_SIZE)
        images = np.transpose(images, (0, 2, 3, 1)).astype(np.float32)
        images = images / 127.5 - 1.0
        labels = np.array(batch[b"labels"], dtype=np.int32)
        image_batches.append(images)
        label_batches.append(labels)

    images = np.concatenate(image_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    return images, labels


def batch_iterator(images, labels, batch_size, drop_last=True):
    indices = np.random.permutation(len(images))
    limit = len(indices)
    if drop_last:
        limit = (limit // batch_size) * batch_size

    for start in range(0, limit, batch_size):
        batch_ids = indices[start : start + batch_size]
        if len(batch_ids) < batch_size and drop_last:
            continue
        yield (
            mx.array(images[batch_ids]),
            mx.array(labels[batch_ids]),
        )


def prompt_to_labels(prompts):
    if isinstance(prompts, str):
        prompts = [prompts]

    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    label_ids = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        match = next((idx for name, idx in name_to_idx.items() if name in prompt_lower), None)
        if match is None:
            valid = ", ".join(CLASS_NAMES)
            raise ValueError(
                f"Could not map prompt '{prompt}' to a CIFAR-10 class. Try one of: {valid}"
            )
        label_ids.append(match)
    return mx.array(np.array(label_ids, dtype=np.int32))


def loss_fn(model, images, labels, schedule):
    batch_size = images.shape[0]
    timesteps = mx.random.randint(0, TIMESTEPS, shape=(batch_size,), dtype=mx.int32)
    keep_mask = mx.random.uniform(shape=(batch_size,)) > COND_DROP_PROB
    x_noisy, true_noise = forward_diffusion(images, timesteps, schedule)
    predicted_noise = model(x_noisy, timesteps, labels=labels, cond_mask=keep_mask)
    return nn.losses.mse_loss(predicted_noise, true_noise)


def show_samples(samples, labels=None, title="Generated", save_path=None, show=True):
    images = np.clip((np.array(samples) * 0.5) + 0.5, 0.0, 1.0)
    count = len(images)
    cols = min(count, 8)
    rows = (count + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(-1)
    for idx, axis in enumerate(axes):
        if idx < count:
            axis.imshow(images[idx])
            if labels is not None:
                label_idx = int(np.array(labels)[idx])
                axis.set_title(CLASS_NAMES[label_idx], fontsize=8)
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def show_forward_process(images, schedule, save_path="forward_cifar10_labels_mlx.png", show=True):
    image = mx.array(images[:1])
    steps = [0, 50, 150, 400, 700, 999]
    fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 2, 2.2))

    for axis, timestep in zip(axes, steps):
        noisy, _ = forward_diffusion(image, mx.array([timestep], dtype=mx.int32), schedule)
        pixels = np.clip(np.array(noisy[0]) * 0.5 + 0.5, 0.0, 1.0)
        axis.imshow(pixels)
        axis.set_title(f"t={timestep}")
        axis.axis("off")

    fig.suptitle("Forward diffusion (image -> noise)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"saved {save_path}")


def sample(
    model,
    schedule,
    labels=None,
    num_samples=None,
    guidance_scale=GUIDANCE_SCALE,
    sample_steps=None,
    ddim_eta=0.0,
):
    model.eval()

    if labels is None:
        if num_samples is None:
            raise ValueError("provide either labels or num_samples")
        label_array = None
        sample_count = num_samples
    elif isinstance(labels, str):
        label_array = prompt_to_labels([labels])
        sample_count = label_array.shape[0]
    elif isinstance(labels, int):
        label_array = mx.array(np.array([labels], dtype=np.int32))
        sample_count = 1
    elif isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], str):
        label_array = prompt_to_labels(labels)
        sample_count = label_array.shape[0]
    elif isinstance(labels, (list, tuple)):
        label_array = mx.array(np.array(labels, dtype=np.int32))
        sample_count = label_array.shape[0]
    else:
        label_array = labels
        sample_count = label_array.shape[0]

    x = mx.random.normal(shape=(sample_count, IMG_SIZE, IMG_SIZE, CHANNELS))
    sampling_timesteps = make_sampling_timesteps(sample_steps)

    for step_idx, timestep in enumerate(sampling_timesteps):
        next_timestep = sampling_timesteps[step_idx + 1] if step_idx + 1 < len(sampling_timesteps) else -1
        t = mx.full((sample_count,), timestep, dtype=mx.int32)

        if label_array is None:
            pred_noise = model(x, t, labels=None)
        else:
            cond_noise = model(x, t, labels=label_array)
            uncond_noise = model(x, t, labels=None)
            pred_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

        alpha_bar_t = schedule.alpha_cumprod[timestep]
        alpha_bar_next = mx.array(1.0) if next_timestep < 0 else schedule.alpha_cumprod[next_timestep]

        pred_x0 = (x - mx.sqrt(1.0 - alpha_bar_t) * pred_noise) / mx.sqrt(alpha_bar_t)

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

    model.train()
    return mx.clip(x, -1.0, 1.0)


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    images, labels = load_cifar10_train(root=args.data_dir, download=not args.no_download)
    schedule = make_schedule()
    model = SmallUNet(
        in_channels=CHANNELS,
        base_channels=args.base_channels,
        time_dim=args.time_embed_dim,
        num_classes=len(CLASS_NAMES),
    )
    if args.resume_from:
        model.load_weights(args.resume_from)
        print(f"loaded checkpoint from {args.resume_from}")
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    total_params = count_params(model.parameters())
    print(f"model parameters: {total_params / 1e6:.2f}M")
    print(f"training images: {len(images)}")

    steps_per_epoch = len(images) // args.batch_size
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch_images, batch_labels in batch_iterator(images, labels, args.batch_size, drop_last=True):
            loss, grads = loss_and_grad_fn(model, batch_images, batch_labels, schedule)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total_loss += float(np.array(loss))

        avg_loss = total_loss / max(steps_per_epoch, 1)
        print(f"epoch {epoch:>2d}/{args.epochs} avg_loss={avg_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == 1:
            preview_labels = mx.array(
                np.arange(args.preview_count, dtype=np.int32) % len(CLASS_NAMES)
            )
            preview = sample(
                model,
                schedule,
                labels=preview_labels,
                guidance_scale=args.guidance_scale,
                sample_steps=args.sample_steps,
                ddim_eta=args.ddim_eta,
            )
            show_samples(
                preview,
                labels=preview_labels,
                title=f"Epoch {epoch}",
                save_path=f"samples_cifar10_labels_epoch{epoch:02d}.png",
                show=not args.no_show,
            )

    model.save_weights(args.weights_path)
    print(f"saved model weights to {args.weights_path}")
    return model, schedule


def build_parser():
    parser = argparse.ArgumentParser(description="Train or sample a CIFAR-10 text-conditional diffusion model in MLX.")
    parser.add_argument("--mode", choices=["train", "sample", "forward"], default="train")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--weights-path", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--time-embed-dim", type=int, default=TIME_EMBED_DIM)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--sample-steps", type=int, default=100, help="Number of reverse diffusion steps to use at sample time.")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="0.0 gives deterministic DDIM sampling; higher values add noise.")
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--final-samples", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=16, help="Used for unconditional sample mode.")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=["airplane", "cat", "dog", "ship"],
        help="Class prompts for conditional sampling.",
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Ignore prompts and generate unconditional samples.",
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        model, schedule = train(args)
        demo_labels = prompt_to_labels(args.prompts)
        demo_samples = sample(
            model,
            schedule,
            labels=demo_labels,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
        )
        show_samples(
            demo_samples,
            labels=demo_labels,
            title="Prompted samples",
            save_path="samples_prompted_mlx.png",
            show=not args.no_show,
        )

        final_samples = sample(
            model,
            schedule,
            num_samples=args.final_samples,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
        )
        show_samples(
            final_samples,
            title="Final Generated CIFAR-10 Images",
            save_path="final_cifar10_labels_mlx.png",
            show=not args.no_show,
        )
        return

    schedule = make_schedule()
    model = SmallUNet(
        in_channels=CHANNELS,
        base_channels=args.base_channels,
        time_dim=args.time_embed_dim,
        num_classes=len(CLASS_NAMES),
    )
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())

    if args.mode == "forward":
        images, _ = load_cifar10_train(root=args.data_dir, download=not args.no_download)
        show_forward_process(images, schedule, show=not args.no_show)
        return

    if not args.unconditional and args.prompts:
        prompt_values = args.prompts[: args.preview_count]
        prompt_labels = prompt_to_labels(prompt_values)
        samples = sample(
            model,
            schedule,
            labels=prompt_labels,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
        )
        show_samples(
            samples,
            labels=prompt_labels,
            title="Prompted samples",
            save_path="samples_prompted_mlx.png",
            show=not args.no_show,
        )
    else:
        samples = sample(
            model,
            schedule,
            num_samples=args.num_samples,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
        )
        show_samples(
            samples,
            title="Unconditional samples",
            save_path="samples_unconditional_mlx.png",
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
