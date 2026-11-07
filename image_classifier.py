"""Fine-tune a ResNet-50 on DATASET/TRAIN + DATASET/VAL and classify images.

The training images are background-removed cutouts (RGBA, mostly transparent) that fill
the frame, while the real test photos (Test_Images) are large phone pictures of small
glassware on a desk, with real backgrounds, portrait orientation and EXIF rotation.
The data pipeline below is built to close that gap: cutouts are zoomed out and
composited onto randomised backgrounds, images are letterboxed instead of squashed,
EXIF orientation is honoured and augmentation is aggressive.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import matplotlib

import image_tool
import labor_glass_model
from labor_glass_model import ImageClassifier_resnet50
from image_transforms import get_image_size, make_transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "DATASET" / "TRAIN"
VAL_DIR = BASE_DIR / "DATASET" / "VAL"
MODEL_PATH = BASE_DIR / "model.pth"
HISTORY_PATH = BASE_DIR / "training_history.json"

BATCH_SIZE = 16
EPOCHS = 30
FREEZE_EPOCHS = 4
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.5
PATIENCE = 8
MIX_PROBABILITY = 0.5
ZOOM_OUT_PROBABILITY = 0.6
ARCHITECTURE = "resnet50"
LEGACY_ARCHITECTURE = "resnet18"


# Letterbox colour: ImageNet mean, so padding is "neutral" after normalisation.


IMAGE_EXTENSIONS = tuple(datasets.folder.IMG_EXTENSIONS)







class ReadableImageFilter:
    """ImageFolder ``is_valid_file`` hook that drops unreadable/corrupt files and remembers them."""

    def __init__(self) -> None:
        self.skipped: list[str] = []

    def __call__(self, path: str) -> bool:
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            return False
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            self.skipped.append(path)
            return False
        return True


class GlasswareDataset(datasets.ImageFolder):
    def __init__(self, root: Path, training: bool) -> None:
        self.filter = ReadableImageFilter()
        super().__init__(str(root), transform=make_transforms(training), is_valid_file=self.filter)
        self.training = training

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if self.training:
            rng = random
        else:
            # Stable per-file background/zoom so validation numbers are comparable across epochs.
            digest = hashlib.md5(path.encode("utf-8")).hexdigest()
            rng = random.Random(int(digest[:8], 16))
        image = image_tool.load_image(Path(path), rng, ZOOM_OUT_PROBABILITY)
        return self.transform(image), target

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for _, target in self.samples:
            counts[target] += 1
        return counts


# -------------------------------------------------------------------------- training


def _set_backbone_trainable(model: ImageClassifier_resnet50, trainable: bool) -> None:
    for parameter in model.backbone_parameters():
        parameter.requires_grad_(trainable)


def _evaluate(model, loader, loss_function, device, num_classes, use_amp):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1
    total = int(confusion.sum().item())
    accuracy = confusion.diag().sum().item() / max(total, 1)
    per_class_recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()
    balanced_accuracy = per_class_recall.mean().item()
    return total_loss / max(total, 1), accuracy, balanced_accuracy, confusion


def train_model(args: argparse.Namespace) -> None:
    if not TRAIN_DIR.is_dir() or not VAL_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected training and validation folders at {TRAIN_DIR} and {VAL_DIR}."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = GlasswareDataset(TRAIN_DIR, training=True)
    val_dataset = GlasswareDataset(VAL_DIR, training=False)
    if not train_dataset.classes:
        raise ValueError(f"No class folders found in {TRAIN_DIR}.")
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Training and validation folders must contain the same class folders. "
            f"Training: {train_dataset.classes}; validation: {val_dataset.classes}"
        )
    skipped = train_dataset.filter.skipped + val_dataset.filter.skipped
    if skipped:
        print(f"Skipped {len(skipped)} unreadable image files:")
        for path in skipped:
            print(f"  {path}")

    classes = train_dataset.classes
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    # Oversample rare classes so every class is seen equally often.
    class_counts = train_dataset.class_counts()
    sample_weights = [1.0 / class_counts[target] for _, target in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = ImageClassifier_resnet50(num_classes, pretrained=True).to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    mixer = v2.RandomChoice([v2.CutMix(num_classes=num_classes), v2.MixUp(num_classes=num_classes, alpha=0.4)])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.backbone_lr, "weight_decay": WEIGHT_DECAY},
            {"params": model.head_parameters(), "lr": args.head_lr, "weight_decay": 1e-4},
        ]
    )
    warmup_epochs = max(args.freeze_epochs, 1)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    best_balanced_accuracy = -1.0
    best_epoch = 0
    best_confusion = None
    epochs_without_improvement = 0

    print(
        f"Training on {device}{' with AMP' if use_amp else ''}; "
        f"{len(train_dataset)} train / {len(val_dataset)} val images; classes: {', '.join(classes)}"
    )
    print(f"Train class counts: {dict(zip(classes, class_counts))}")

    for epoch in range(args.epochs):
        started = time.time()
        backbone_trainable = epoch >= args.freeze_epochs
        _set_backbone_trainable(model, backbone_trainable)
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = labels
            if not args.no_mix and random.random() < MIX_PROBABILITY:
                images, targets = mixer(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = loss_function(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_seen += images.size(0)

        val_loss, val_accuracy, val_balanced_accuracy, confusion = _evaluate(
            model, val_loader, loss_function, device, num_classes, use_amp
        )
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()

        train_loss = total_loss / max(train_seen, 1)
        train_accuracy = train_correct / max(train_seen, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": val_balanced_accuracy,
                "backbone_trainable": backbone_trainable,
                "learning_rates": learning_rates,
                "seconds": time.time() - started,
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} ({time.time() - started:.0f}s) - "
            f"loss: {train_loss:.4f} - train accuracy: {train_accuracy:.1%} - "
            f"val loss: {val_loss:.4f} - val accuracy: {val_accuracy:.1%} - "
            f"val balanced accuracy: {val_balanced_accuracy:.1%}"
            f"{' (backbone frozen)' if not backbone_trainable else ''}"
        )

        if val_balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = val_balanced_accuracy
            best_epoch = epoch + 1
            best_confusion = confusion.tolist()
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": get_image_size(),
                    "preprocessing": "letterbox",
                    "epoch": best_epoch,
                    "validation_accuracy": val_accuracy,
                    "balanced_validation_accuracy": val_balanced_accuracy,
                },
                args.model,
            )
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"No improvement for {args.patience} epochs; stopping early.")
                break

    print(f"Best balanced validation accuracy: {best_balanced_accuracy:.1%} (epoch {best_epoch})")
    print(f"Model saved to {args.model}")

    HISTORY_PATH.write_text(
        json.dumps(
            {
                "classes": classes,
                "best_epoch": best_epoch,
                "best_balanced_validation_accuracy": best_balanced_accuracy,
                "confusion_matrix_best_epoch": best_confusion,
                "skipped_files": skipped,
                "epochs": history,
            },
            indent=2,
        )
    )
    _plot_history(history)


def _plot_history(history: list[dict]) -> None:
    epochs = [entry["epoch"] for entry in history]
    plt.figure()
    plt.plot(epochs, [e["train_loss"] for e in history], marker="o", label="Training")
    plt.plot(epochs, [e["val_loss"] for e in history], marker="o", label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(BASE_DIR / "training_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, [e["train_accuracy"] for e in history], marker="o", label="Training")
    plt.plot(epochs, [e["val_accuracy"] for e in history], marker="o", label="Validation")
    plt.plot(
        epochs,
        [e["val_balanced_accuracy"] for e in history],
        marker="s",
        linestyle="--",
        label="Validation (balanced)",
    )
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(BASE_DIR / "training_accuracy.png")
    plt.close()




def classify_image(image_path: Path, model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = labor_glass_model.load_model(model_path, device)
    probabilities = labor_glass_model.predict_probabilities(model, image_path, device)
    predicted_index = int(probabilities.argmax().item())
    print(
        f"Prediction: {classes[predicted_index]} "
        f"(confidence: {probabilities[predicted_index].item():.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an image classifier or classify one image."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-train", action="store_true", help="train and save a new model")
    mode.add_argument("-image", type=Path, help="image file to classify")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="checkpoint path to write/read")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument(
        "--freeze-epochs", type=int, default=FREEZE_EPOCHS,
        help="epochs to train only the classifier head before unfreezing the backbone",
    )
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--head-lr", type=float, default=HEAD_LR)
    parser.add_argument(
        "--patience", type=int, default=PATIENCE,
        help="stop when balanced validation accuracy has not improved for this many epochs (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    parser.add_argument("--no-mix", action="store_true", help="disable CutMix/MixUp augmentation")
    args = parser.parse_args()

    if args.train:
        train_model(args)
    else:
        classify_image(args.image, args.model)


if __name__ == "__main__":
    main()