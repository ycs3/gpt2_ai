import argparse
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
NUM_FRAMES = 3
CHANNELS = 3
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
BASE_CHANNELS = 48
WEIGHTS_PATH = "frame_interpolation_ucf101_mlx.safetensors"
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
        train_paths = filter_and_limit_paths(train_candidates, class_to_idx, max_train_videos_per_class)
        test_paths = filter_and_limit_paths(test_candidates, class_to_idx, max_test_videos_per_class)
    else:
        discovered = discover_video_paths(videos_dir)
        class_to_idx = build_class_mapping(discovered, selected_classes)
        discovered = filter_and_limit_paths(discovered, class_to_idx)
        train_candidates, test_candidates = split_samples(discovered, seed=seed)
        train_paths = filter_and_limit_paths(train_candidates, class_to_idx, max_train_videos_per_class)
        test_paths = filter_and_limit_paths(test_candidates, class_to_idx, max_test_videos_per_class)

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


def make_interpolation_triplets(clips):
    left = clips[:, 0]
    middle = clips[:, 1]
    right = clips[:, 2]
    inputs = np.concatenate([left, right], axis=-1)
    return inputs, middle, left, right


def batch_iterator(inputs, targets, labels, batch_size, shuffle=True, drop_last=False):
    indices = np.arange(len(inputs))
    if shuffle:
        indices = np.random.permutation(indices)
    limit = len(indices)
    if drop_last:
        limit = (limit // batch_size) * batch_size
    for start in range(0, limit, batch_size):
        batch_ids = indices[start : start + batch_size]
        if len(batch_ids) < batch_size and drop_last:
            continue
        yield mx.array(inputs[batch_ids]), mx.array(targets[batch_ids]), mx.array(labels[batch_ids])


class FrameInterpolator(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS):
        super().__init__()
        self.enc1 = nn.Conv2d(CHANNELS * 2, base_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.mid1 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.mid2 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(base_channels, CHANNELS, kernel_size=3, padding=1)

    def __call__(self, x):
        h1 = nn.silu(self.enc1(x))
        h2 = nn.silu(self.enc2(h1))
        h3 = nn.silu(self.enc3(h2))

        h = nn.silu(self.mid1(h3))
        h = nn.silu(self.mid2(h))

        h = nn.silu(self.up2(h))
        h = mx.concatenate([h, h2], axis=-1)
        h = nn.silu(self.dec2(h))
        h = nn.silu(self.up1(h))
        h = mx.concatenate([h, h1], axis=-1)
        h = nn.silu(self.dec1(h))
        return mx.tanh(self.out(h))


def loss_fn(model, inputs, targets):
    predictions = model(inputs)
    return nn.losses.mse_loss(predictions, targets)


def evaluate(model, inputs, targets, labels, batch_size):
    total_loss = 0.0
    total_seen = 0
    for batch_inputs, batch_targets, _ in batch_iterator(
        inputs,
        targets,
        labels,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    ):
        predictions = model(batch_inputs)
        loss = nn.losses.mse_loss(predictions, batch_targets)
        mx.eval(loss)
        batch_size_actual = int(batch_inputs.shape[0])
        total_loss += float(np.array(loss)) * batch_size_actual
        total_seen += batch_size_actual
    return total_loss / max(total_seen, 1)


def show_interpolations(left_frames, target_frames, predicted_frames, right_frames, save_path=None, show=True, title="Frame interpolation"):
    left_frames = np.clip((np.array(left_frames) * 0.5) + 0.5, 0.0, 1.0)
    target_frames = np.clip((np.array(target_frames) * 0.5) + 0.5, 0.0, 1.0)
    predicted_frames = np.clip((np.array(predicted_frames) * 0.5) + 0.5, 0.0, 1.0)
    right_frames = np.clip((np.array(right_frames) * 0.5) + 0.5, 0.0, 1.0)

    count = len(left_frames)
    fig, axes = plt.subplots(count, 4, figsize=(6.4, count * 1.8))
    axes = np.array(axes).reshape(count, 4)

    for row in range(count):
        axes[row, 0].imshow(left_frames[row])
        axes[row, 0].set_title("left", fontsize=8)
        axes[row, 1].imshow(target_frames[row])
        axes[row, 1].set_title("target", fontsize=8)
        axes[row, 2].imshow(predicted_frames[row])
        axes[row, 2].set_title("pred", fontsize=8)
        axes[row, 3].imshow(right_frames[row])
        axes[row, 3].set_title("right", fontsize=8)
        for col in range(4):
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
    print(f"loading {len(test_paths)} test videos")
    test_clips, test_labels = load_video_dataset(test_paths, test_labels, args.num_frames, args.image_size)
    return train_clips, train_labels, test_clips, test_labels


def train(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_clips, train_labels, test_clips, test_labels = load_datasets(args)
    train_inputs, train_targets, train_left, train_right = make_interpolation_triplets(train_clips)
    test_inputs, test_targets, test_left, test_right = make_interpolation_triplets(test_clips)

    model = FrameInterpolator(base_channels=args.base_channels)
    if args.resume_from:
        model.load_weights(args.resume_from)
        print(f"loaded checkpoint from {args.resume_from}")
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(f"model parameters: {count_params(model.parameters()) / 1e6:.2f}M")
    print(f"training triplets: {len(train_inputs)}")
    print(f"test triplets: {len(test_inputs)}")

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        seen_steps = 0
        for batch_inputs, batch_targets, _ in batch_iterator(
            train_inputs,
            train_targets,
            train_labels,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        ):
            loss, grads = loss_and_grad_fn(model, batch_inputs, batch_targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            running_loss += float(np.array(loss))
            seen_steps += 1

        train_loss = running_loss / max(seen_steps, 1)
        test_loss = evaluate(model, test_inputs, test_targets, test_labels, args.eval_batch_size)
        print(f"epoch {epoch:>2d}/{args.epochs} train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

        if epoch % args.preview_every == 0 or epoch == 1:
            preview_count = min(args.preview_count, len(test_inputs))
            preview_inputs = mx.array(test_inputs[:preview_count])
            predictions = model(preview_inputs)
            mx.eval(predictions)
            show_interpolations(
                test_left[:preview_count],
                test_targets[:preview_count],
                predictions,
                test_right[:preview_count],
                save_path=f"frame_interpolation_epoch{epoch:02d}.png",
                show=not args.no_show,
                title=f"Epoch {epoch}",
            )

    model.save_weights(args.weights_path)
    print(f"saved model weights to {args.weights_path}")
    return model, test_inputs, test_targets, test_left, test_right


def build_parser():
    parser = argparse.ArgumentParser(description="Train or run a small UCF101 frame interpolation model in MLX.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
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
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=IMG_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        model, test_inputs, test_targets, test_left, test_right = train(args)
        preview_count = min(args.preview_count, len(test_inputs))
        predictions = model(mx.array(test_inputs[:preview_count]))
        mx.eval(predictions)
        show_interpolations(
            test_left[:preview_count],
            test_targets[:preview_count],
            predictions,
            test_right[:preview_count],
            save_path="frame_interpolation_predictions.png",
            show=not args.no_show,
            title="Post-training interpolation",
        )
        return

    _, _, test_clips, test_labels = load_datasets(args)
    test_inputs, test_targets, test_left, test_right = make_interpolation_triplets(test_clips)
    model = FrameInterpolator(base_channels=args.base_channels)
    model.load_weights(args.weights_path)
    mx.eval(model.parameters())
    preview_count = min(args.preview_count, len(test_inputs))
    predictions = model(mx.array(test_inputs[:preview_count]))
    mx.eval(predictions)
    show_interpolations(
        test_left[:preview_count],
        test_targets[:preview_count],
        predictions,
        test_right[:preview_count],
        save_path="frame_interpolation_predictions.png",
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
