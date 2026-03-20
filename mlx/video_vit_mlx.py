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


IMG_SIZE = 96
NUM_FRAMES = 8
PATCH_SIZE = 16
EMBED_DIM = 192
DEPTH = 6
NUM_HEADS = 6
MLP_RATIO = 4
EPOCHS = 12
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WEIGHTS_PATH = "video_vit_ucf101_mlx.safetensors"
DEFAULT_CLASSES = ["Basketball", "Biking", "HorseRiding", "Surfing", "Typing"]


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
        discovered = filter_and_limit_paths(discovered, class_to_idx, max_videos_per_class=None)
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

    if not train_paths or not test_paths:
        raise ValueError("Dataset split is empty. Check the dataset path, class list, or split files.")

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
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr.decode('utf-8', errors='ignore')}")

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


def random_horizontal_flip(clips):
    flipped = clips.copy()
    mask = np.random.rand(clips.shape[0]) < 0.5
    flipped[mask] = flipped[mask, :, :, ::-1, :]
    return flipped


def batch_iterator(clips, labels, batch_size, shuffle=True, augment=False, drop_last=False):
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
        batch_clips = clips[batch_ids]
        if augment:
            batch_clips = random_horizontal_flip(batch_clips)
        yield mx.array(batch_clips), mx.array(labels[batch_ids])


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def __call__(self, x):
        batch_size, num_frames, height, width, channels = x.shape
        x = x.reshape(batch_size * num_frames, height, width, channels)
        x = self.proj(x)
        _, patch_h, patch_w, embed_dim = x.shape
        x = x.reshape(batch_size, num_frames, patch_h * patch_w, embed_dim)
        return x


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


class FactorizedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.spatial_norm1 = nn.LayerNorm(embed_dim)
        self.spatial_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.spatial_norm2 = nn.LayerNorm(embed_dim)
        self.spatial_mlp = MLP(embed_dim, mlp_ratio)

        self.temporal_norm1 = nn.LayerNorm(embed_dim)
        self.temporal_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.temporal_norm2 = nn.LayerNorm(embed_dim)
        self.temporal_mlp = MLP(embed_dim, mlp_ratio)

    def __call__(self, x):
        batch_size, num_frames, num_patches, embed_dim = x.shape

        spatial = x.reshape(batch_size * num_frames, num_patches, embed_dim)
        spatial = spatial + self.spatial_attn(self.spatial_norm1(spatial))
        spatial = spatial + self.spatial_mlp(self.spatial_norm2(spatial))
        x = spatial.reshape(batch_size, num_frames, num_patches, embed_dim)

        temporal = x.transpose(0, 2, 1, 3).reshape(batch_size * num_patches, num_frames, embed_dim)
        temporal = temporal + self.temporal_attn(self.temporal_norm1(temporal))
        temporal = temporal + self.temporal_mlp(self.temporal_norm2(temporal))
        return temporal.reshape(batch_size, num_patches, num_frames, embed_dim).transpose(0, 2, 1, 3)


class VideoVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=IMG_SIZE,
        num_frames=NUM_FRAMES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=5,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.spatial_pos_embed = mx.zeros((1, 1, num_patches, embed_dim))
        self.temporal_pos_embed = mx.zeros((1, num_frames, 1, embed_dim))
        self.blocks = [FactorizedTransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = x + self.spatial_pos_embed + self.temporal_pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = mx.mean(x, axis=(1, 2))
        return self.head(x)


def cross_entropy_loss(logits, labels):
    shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    label_log_probs = mx.take_along_axis(log_probs, labels.reshape(-1, 1), axis=-1)
    return -mx.mean(label_log_probs)


def loss_fn(model, clips, labels):
    return cross_entropy_loss(model(clips), labels)


def evaluate(model, clips, labels, batch_size):
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_clips, batch_labels in batch_iterator(
        clips,
        labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        drop_last=False,
    ):
        logits = model(batch_clips)
        loss = cross_entropy_loss(logits, batch_labels)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(loss, predictions)
        batch_size_actual = int(batch_labels.shape[0])
        total_loss += float(np.array(loss)) * batch_size_actual
        total_correct += int(np.sum(np.array(predictions) == np.array(batch_labels)))
        total_seen += batch_size_actual
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def show_predictions(clips, labels, predictions, class_names, save_path=None, show=True):
    clips_np = np.clip((np.array(clips) * 0.5) + 0.5, 0.0, 1.0)
    labels_np = np.array(labels)
    predictions_np = np.array(predictions)
    count = len(clips_np)
    fig, axes = plt.subplots(count, clips_np.shape[1], figsize=(clips_np.shape[1] * 1.6, count * 1.8))
    axes = np.array(axes).reshape(count, clips_np.shape[1])

    for row in range(count):
        truth = class_names[int(labels_np[row])]
        pred = class_names[int(predictions_np[row])]
        for col in range(clips_np.shape[1]):
            axes[row, col].imshow(clips_np[row, col])
            if col == 0:
                axes[row, col].set_title(f"t: {truth}\np: {pred}", fontsize=8)
            axes[row, col].axis("off")

    fig.suptitle("Video ViT predictions", fontsize=13)
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
    print(f"loading {len(test_paths)} test videos")
    test_clips, test_labels = load_video_dataset(test_paths, test_labels, args.num_frames, args.image_size)
    return train_clips, train_labels, test_clips, test_labels, class_names


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_clips, train_labels, test_clips, test_labels, class_names = load_datasets(args)
    model = VideoVisionTransformer(
        image_size=args.image_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=len(class_names),
    )
    if args.resume_from:
        model.load_weights(args.resume_from)
        print(f"loaded checkpoint from {args.resume_from}")
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(f"model parameters: {count_params(model.parameters()) / 1e6:.2f}M")
    print(f"training clips: {len(train_clips)}")
    print(f"test clips: {len(test_clips)}")

    steps_per_epoch = math.ceil(len(train_clips) / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_clips, batch_labels in batch_iterator(
            train_clips,
            train_labels,
            batch_size=args.batch_size,
            shuffle=True,
            augment=not args.no_augment,
            drop_last=False,
        ):
            loss, grads = loss_and_grad_fn(model, batch_clips, batch_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            running_loss += float(np.array(loss))

        train_loss = running_loss / max(steps_per_epoch, 1)
        test_loss, test_accuracy = evaluate(model, test_clips, test_labels, args.eval_batch_size)
        print(
            f"epoch {epoch:>2d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_accuracy * 100:.2f}%"
        )

    model.save_weights(args.weights_path)
    print(f"saved model weights to {args.weights_path}")
    return model, test_clips, test_labels, class_names


def run_eval(args):
    _, _, test_clips, test_labels, class_names = load_datasets(args)
    model = VideoVisionTransformer(
        image_size=args.image_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=len(class_names),
    )
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())
    test_loss, test_accuracy = evaluate(model, test_clips, test_labels, args.eval_batch_size)
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy * 100:.2f}%")


def run_predict(args):
    np.random.seed(args.seed)
    _, _, test_clips, test_labels, class_names = load_datasets(args)
    model = VideoVisionTransformer(
        image_size=args.image_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=len(class_names),
    )
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())

    sample_count = min(args.num_samples, len(test_clips))
    sample_ids = np.random.choice(len(test_clips), size=sample_count, replace=False)
    sample_clips = mx.array(test_clips[sample_ids])
    sample_labels = mx.array(test_labels[sample_ids])
    logits = model(sample_clips)
    predictions = mx.argmax(logits, axis=-1)
    mx.eval(logits, predictions)

    accuracy = float(np.mean(np.array(predictions) == np.array(sample_labels)))
    print(f"prediction batch accuracy={accuracy * 100:.2f}%")
    show_predictions(
        sample_clips,
        sample_labels,
        predictions,
        class_names=class_names,
        save_path=args.predictions_path,
        show=not args.no_show,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a small UCF101 video Vision Transformer in MLX."
    )
    parser.add_argument("--mode", choices=["train", "eval", "predict"], default="train")
    parser.add_argument("--videos-dir", type=str, required=True)
    parser.add_argument("--train-split-file", type=str, default=None)
    parser.add_argument("--test-split-file", type=str, default=None)
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--max-train-videos-per-class", type=int, default=80)
    parser.add_argument("--max-test-videos-per-class", type=int, default=20)
    parser.add_argument("--weights-path", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=IMG_SIZE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--mlp-ratio", type=int, default=MLP_RATIO)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--predictions-path", type=str, default="video_vit_predictions.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        model, test_clips, test_labels, class_names = train(args)
        preview_count = min(args.num_samples, len(test_clips))
        preview_clips = mx.array(test_clips[:preview_count])
        preview_labels = mx.array(test_labels[:preview_count])
        logits = model(preview_clips)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(logits, predictions)
        show_predictions(
            preview_clips,
            preview_labels,
            predictions,
            class_names=class_names,
            save_path=args.predictions_path,
            show=not args.no_show,
        )
        return

    if args.mode == "eval":
        run_eval(args)
        return

    run_predict(args)


if __name__ == "__main__":
    main()
