import argparse
import math
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
NUM_CLASSES = 10
PATCH_SIZE = 4
EMBED_DIM = 192
DEPTH = 6
NUM_HEADS = 6
MLP_RATIO = 4
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WEIGHTS_PATH = "cifar10_vit_mlx.safetensors"
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


def gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


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


def load_cifar10_split(root="./data", train=True, download=True):
    root_path = Path(root)
    extracted_dir = root_path / "cifar-10-batches-py"
    if not extracted_dir.exists():
        if not download:
            raise FileNotFoundError(f"missing CIFAR-10 data at {extracted_dir}")
        extracted_dir = ensure_cifar10(root_path)

    batch_names = [f"data_batch_{idx}" for idx in range(1, 6)] if train else ["test_batch"]
    image_batches = []
    label_batches = []
    for batch_name in batch_names:
        batch_path = extracted_dir / batch_name
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


def random_flip(images):
    flipped = images.copy()
    mask = np.random.rand(images.shape[0]) < 0.5
    flipped[mask] = flipped[mask, :, ::-1, :]
    return flipped


def batch_iterator(images, labels, batch_size, shuffle=True, augment=False, drop_last=False):
    indices = np.arange(len(images))
    if shuffle:
        indices = np.random.permutation(indices)

    limit = len(indices)
    if drop_last:
        limit = (limit // batch_size) * batch_size

    for start in range(0, limit, batch_size):
        batch_ids = indices[start : start + batch_size]
        if len(batch_ids) < batch_size and drop_last:
            continue
        batch_images = images[batch_ids]
        if augment:
            batch_images = random_flip(batch_images)
        yield mx.array(batch_images), mx.array(labels[batch_ids])


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            CHANNELS,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def __call__(self, x):
        x = self.proj(x)
        batch_size, height, width, channels = x.shape
        return x.reshape(batch_size, height * width, channels)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = stable_softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        return self.out(out)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid_size = image_size // patch_size
        self.num_patches = grid_size * grid_size
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, self.num_patches + 1, embed_dim))
        self.blocks = [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def __call__(self, x):
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        cls_tokens = mx.broadcast_to(self.cls_token, (batch_size, 1, self.cls_token.shape[-1]))
        x = mx.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def cross_entropy_loss(logits, labels):
    shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    label_log_probs = mx.take_along_axis(log_probs, labels.reshape(-1, 1), axis=-1)
    return -mx.mean(label_log_probs)


def loss_fn(model, images, labels):
    logits = model(images)
    return cross_entropy_loss(logits, labels)


def evaluate(model, images, labels, batch_size):
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_images, batch_labels in batch_iterator(
        images,
        labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        drop_last=False,
    ):
        logits = model(batch_images)
        loss = cross_entropy_loss(logits, batch_labels)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(loss, predictions)
        batch_size_actual = int(batch_labels.shape[0])
        total_loss += float(np.array(loss)) * batch_size_actual
        total_correct += int(np.sum(np.array(predictions) == np.array(batch_labels)))
        total_seen += batch_size_actual

    avg_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    return avg_loss, accuracy


def show_predictions(images, labels, predictions, save_path=None, show=True, title="ViT predictions"):
    pixels = np.clip((np.array(images) * 0.5) + 0.5, 0.0, 1.0)
    labels = np.array(labels)
    predictions = np.array(predictions)
    count = len(pixels)
    cols = min(count, 4)
    rows = (count + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for idx, axis in enumerate(axes):
        if idx < count:
            axis.imshow(pixels[idx])
            truth = CLASS_NAMES[int(labels[idx])]
            pred = CLASS_NAMES[int(predictions[idx])]
            axis.set_title(f"t: {truth}\np: {pred}", fontsize=8)
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


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_images, train_labels = load_cifar10_split(
        root=args.data_dir,
        train=True,
        download=not args.no_download,
    )
    test_images, test_labels = load_cifar10_split(
        root=args.data_dir,
        train=False,
        download=not args.no_download,
    )

    model = VisionTransformer(
        image_size=IMG_SIZE,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=len(CLASS_NAMES),
    )
    if args.resume_from:
        model.load_weights(args.resume_from)
        print(f"loaded checkpoint from {args.resume_from}")
    mx.eval(model.parameters())

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    total_params = count_params(model.parameters())
    print(f"model parameters: {total_params / 1e6:.2f}M")
    print(f"training images: {len(train_images)}")
    print(f"test images: {len(test_images)}")

    steps_per_epoch = math.ceil(len(train_images) / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_images, batch_labels in batch_iterator(
            train_images,
            train_labels,
            batch_size=args.batch_size,
            shuffle=True,
            augment=not args.no_augment,
            drop_last=False,
        ):
            loss, grads = loss_and_grad_fn(model, batch_images, batch_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            running_loss += float(np.array(loss))

        train_loss = running_loss / max(steps_per_epoch, 1)
        test_loss, test_accuracy = evaluate(model, test_images, test_labels, args.eval_batch_size)
        print(
            f"epoch {epoch:>2d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_accuracy * 100:.2f}%"
        )

    model.save_weights(args.weights_path)
    print(f"saved model weights to {args.weights_path}")
    return model, test_images, test_labels


def run_eval(args):
    images, labels = load_cifar10_split(
        root=args.data_dir,
        train=False,
        download=not args.no_download,
    )
    model = VisionTransformer(
        image_size=IMG_SIZE,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=len(CLASS_NAMES),
    )
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())
    test_loss, test_accuracy = evaluate(model, images, labels, args.eval_batch_size)
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy * 100:.2f}%")


def run_predict(args):
    np.random.seed(args.seed)
    images, labels = load_cifar10_split(
        root=args.data_dir,
        train=False,
        download=not args.no_download,
    )
    model = VisionTransformer(
        image_size=IMG_SIZE,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=len(CLASS_NAMES),
    )
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())

    sample_count = min(args.num_samples, len(images))
    sample_ids = np.random.choice(len(images), size=sample_count, replace=False)
    sample_images = mx.array(images[sample_ids])
    sample_labels = mx.array(labels[sample_ids])
    logits = model(sample_images)
    predictions = mx.argmax(logits, axis=-1)
    mx.eval(logits, predictions)

    accuracy = float(np.mean(np.array(predictions) == np.array(sample_labels)))
    print(f"prediction batch accuracy={accuracy * 100:.2f}%")
    show_predictions(
        sample_images,
        sample_labels,
        predictions,
        save_path=args.predictions_path,
        show=not args.no_show,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate a CIFAR-10 Vision Transformer in MLX.")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], default="train")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--weights-path", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--mlp-ratio", type=int, default=MLP_RATIO)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--predictions-path", type=str, default="cifar10_vit_predictions.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        model, test_images, test_labels = train(args)
        preview_count = min(args.num_samples, len(test_images))
        preview_images = mx.array(test_images[:preview_count])
        preview_labels = mx.array(test_labels[:preview_count])
        logits = model(preview_images)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(logits, predictions)
        show_predictions(
            preview_images,
            preview_labels,
            predictions,
            save_path=args.predictions_path,
            show=not args.no_show,
            title="Post-training CIFAR-10 ViT predictions",
        )
        return

    if args.mode == "eval":
        run_eval(args)
        return

    run_predict(args)


if __name__ == "__main__":
    main()
