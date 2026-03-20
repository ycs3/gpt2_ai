import argparse
import pickle
import tarfile
import urllib.request
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


IMG_SIZE = 32
CHANNELS = 3
EPOCHS = 50
BATCH_SIZE = 128
LATENT_DIM = 128
BASE_CHANNELS = 64
LEARNING_RATE = 2e-4
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
GENERATOR_PATH = "cifar10_gan_mlx_generator.safetensors"
DISCRIMINATOR_PATH = "cifar10_gan_mlx_discriminator.safetensors"
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
        yield mx.array(images[batch_ids]), mx.array(labels[batch_ids])


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


def softplus(x):
    return mx.maximum(x, 0.0) + mx.log1p(mx.exp(-mx.abs(x)))


def bce_with_logits(logits, targets):
    return mx.mean(softplus(logits) - targets * logits)


def leaky_relu(x, slope=0.2):
    return mx.maximum(x, 0.0) + slope * mx.minimum(x, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, base_channels=BASE_CHANNELS, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, 4 * 4 * base_channels * 4)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(base_channels, CHANNELS, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, base_channels * 2, pytorch_compatible=True)
        self.norm2 = nn.GroupNorm(8, base_channels, pytorch_compatible=True)

    def __call__(self, z, labels=None):
        if labels is None:
            class_cond = mx.zeros((z.shape[0], self.latent_dim), dtype=z.dtype)
        else:
            class_cond = self.class_embed(labels)

        h = mx.concatenate([z, class_cond], axis=-1)
        h = self.fc(h).reshape(z.shape[0], 4, 4, -1)
        h = nn.silu(h)
        h = self.up1(h)
        h = self.norm1(nn.silu(h))
        h = self.up2(h)
        h = self.norm2(nn.silu(h))
        h = self.up3(h)
        return mx.tanh(h)


class Discriminator(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, IMG_SIZE * IMG_SIZE)
        self.conv1 = nn.Conv2d(CHANNELS + 1, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, base_channels * 2, pytorch_compatible=True)
        self.norm3 = nn.GroupNorm(8, base_channels * 4, pytorch_compatible=True)
        self.fc = nn.Linear(4 * 4 * base_channels * 4, 1)

    def __call__(self, images, labels=None):
        if labels is None:
            class_map = mx.zeros((images.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=images.dtype)
        else:
            class_map = self.class_embed(labels).reshape(images.shape[0], IMG_SIZE, IMG_SIZE, 1)

        x = mx.concatenate([images, class_map], axis=-1)
        x = leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.norm2(leaky_relu(x))
        x = self.conv3(x)
        x = self.norm3(leaky_relu(x))
        x = x.reshape(images.shape[0], -1)
        return self.fc(x).reshape(-1)


def discriminator_loss(discriminator, generator, real_images, real_labels, latent_dim):
    batch_size = real_images.shape[0]
    noise = mx.random.normal(shape=(batch_size, latent_dim))
    fake_images = generator(noise, real_labels)
    real_logits = discriminator(real_images, real_labels)
    fake_logits = discriminator(fake_images, real_labels)
    real_loss = bce_with_logits(real_logits, mx.ones_like(real_logits) * 0.9)
    fake_loss = bce_with_logits(fake_logits, mx.zeros_like(fake_logits))
    return real_loss + fake_loss


def generator_loss(generator, discriminator, labels, latent_dim):
    batch_size = labels.shape[0]
    noise = mx.random.normal(shape=(batch_size, latent_dim))
    fake_images = generator(noise, labels)
    fake_logits = discriminator(fake_images, labels)
    return bce_with_logits(fake_logits, mx.ones_like(fake_logits))


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


def sample(generator, labels=None, num_samples=None, latent_dim=LATENT_DIM):
    generator.eval()

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

    noise = mx.random.normal(shape=(sample_count, latent_dim))
    samples = generator(noise, label_array)
    mx.eval(samples)
    generator.train()
    return mx.clip(samples, -1.0, 1.0)


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    images, labels = load_cifar10_train(root=args.data_dir, download=not args.no_download)
    generator = Generator(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        num_classes=len(CLASS_NAMES),
    )
    discriminator = Discriminator(
        base_channels=args.base_channels,
        num_classes=len(CLASS_NAMES),
    )

    if args.resume_generator:
        generator.load_weights(args.resume_generator)
        print(f"loaded generator checkpoint from {args.resume_generator}")
    if args.resume_discriminator:
        discriminator.load_weights(args.resume_discriminator)
        print(f"loaded discriminator checkpoint from {args.resume_discriminator}")

    mx.eval(generator.parameters(), discriminator.parameters())

    generator_optimizer = optim.Adam(
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    discriminator_optimizer = optim.Adam(
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    discriminator_loss_and_grad = nn.value_and_grad(discriminator, discriminator_loss)
    generator_loss_and_grad = nn.value_and_grad(generator, generator_loss)

    gen_params = count_params(generator.parameters())
    disc_params = count_params(discriminator.parameters())
    print(f"generator parameters: {gen_params / 1e6:.2f}M")
    print(f"discriminator parameters: {disc_params / 1e6:.2f}M")
    print(f"training images: {len(images)}")

    steps_per_epoch = len(images) // args.batch_size
    for epoch in range(1, args.epochs + 1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        for batch_images, batch_labels in batch_iterator(images, labels, args.batch_size, drop_last=True):
            d_loss, d_grads = discriminator_loss_and_grad(
                discriminator,
                generator,
                batch_images,
                batch_labels,
                args.latent_dim,
            )
            discriminator_optimizer.update(discriminator, d_grads)
            mx.eval(discriminator.parameters(), discriminator_optimizer.state, d_loss)
            total_d_loss += float(np.array(d_loss))

            g_loss, g_grads = generator_loss_and_grad(
                generator,
                discriminator,
                batch_labels,
                args.latent_dim,
            )
            generator_optimizer.update(generator, g_grads)
            mx.eval(generator.parameters(), generator_optimizer.state, g_loss)
            total_g_loss += float(np.array(g_loss))

        avg_d_loss = total_d_loss / max(steps_per_epoch, 1)
        avg_g_loss = total_g_loss / max(steps_per_epoch, 1)
        print(
            f"epoch {epoch:>2d}/{args.epochs} "
            f"d_loss={avg_d_loss:.4f} "
            f"g_loss={avg_g_loss:.4f}"
        )

        if epoch % args.sample_every == 0 or epoch == 1:
            preview_labels = mx.array(
                np.arange(args.preview_count, dtype=np.int32) % len(CLASS_NAMES)
            )
            preview = sample(generator, labels=preview_labels, latent_dim=args.latent_dim)
            show_samples(
                preview,
                labels=preview_labels,
                title=f"Epoch {epoch}",
                save_path=f"samples_cifar10_gan_epoch{epoch:02d}.png",
                show=not args.no_show,
            )

    generator.save_weights(args.generator_path)
    discriminator.save_weights(args.discriminator_path)
    print(f"saved generator weights to {args.generator_path}")
    print(f"saved discriminator weights to {args.discriminator_path}")
    return generator


def build_parser():
    parser = argparse.ArgumentParser(description="Train or sample a CIFAR-10 GAN in MLX.")
    parser.add_argument("--mode", choices=["train", "sample"], default="train")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--generator-path", type=str, default=GENERATOR_PATH)
    parser.add_argument("--discriminator-path", type=str, default=DISCRIMINATOR_PATH)
    parser.add_argument("--resume-generator", type=str, default=None)
    parser.add_argument("--resume-discriminator", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--beta1", type=float, default=ADAM_BETA1)
    parser.add_argument("--beta2", type=float, default=ADAM_BETA2)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--final-samples", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=16)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        generator = train(args)
        demo_labels = prompt_to_labels(args.prompts)
        demo_samples = sample(generator, labels=demo_labels, latent_dim=args.latent_dim)
        show_samples(
            demo_samples,
            labels=demo_labels,
            title="Prompted samples",
            save_path="samples_prompted_gan_mlx.png",
            show=not args.no_show,
        )

        final_samples = sample(
            generator,
            num_samples=args.final_samples,
            latent_dim=args.latent_dim,
        )
        show_samples(
            final_samples,
            title="Final Generated CIFAR-10 Images",
            save_path="final_cifar10_gan_mlx.png",
            show=not args.no_show,
        )
        return

    generator = Generator(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        num_classes=len(CLASS_NAMES),
    )
    generator.load_weights(args.generator_path)
    mx.eval(generator.parameters())

    if not args.unconditional and args.prompts:
        prompt_values = args.prompts[: args.preview_count]
        prompt_labels = prompt_to_labels(prompt_values)
        samples = sample(generator, labels=prompt_labels, latent_dim=args.latent_dim)
        show_samples(
            samples,
            labels=prompt_labels,
            title="Prompted samples",
            save_path="samples_prompted_gan_mlx.png",
            show=not args.no_show,
        )
    else:
        samples = sample(
            generator,
            num_samples=args.num_samples,
            latent_dim=args.latent_dim,
        )
        show_samples(
            samples,
            title="Unconditional samples",
            save_path="samples_unconditional_gan_mlx.png",
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
