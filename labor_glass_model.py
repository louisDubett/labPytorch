import torch
import torch.nn as nn
from torchvision import models

from pathlib import Path

from image_tool import load_image
from image_transforms import make_transforms


class ImageClassifier_resnet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.3) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.network = models.resnet50(weights=weights)
        self.network.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.network.fc.in_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def backbone_parameters(self):
        return [p for n, p in self.network.named_parameters() if not n.startswith("fc.")]

    def head_parameters(self):
        return list(self.network.fc.parameters())

    def load_checkpoint_state(self, state_dict: dict) -> None:
        # Checkpoints written before the Dropout head stored the linear layer as "fc.*".
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("network.fc.") and not key.startswith("network.fc.1."):
                key = key.replace("network.fc.", "network.fc.1.", 1)
            remapped[key] = value
        self.load_state_dict(remapped)


class ImageClassifier_Resnet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.network = models.resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class ImageClassifier_vgg19(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        self.network = models.vgg19(weights=weights)
        self.network.classifier[6] = nn.Linear(
            self.network.classifier[6].in_features, num_classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

class ImageClassifier_raw(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

# ------------------------------------------------------------------------- inference

def predict_probabilities(
        model: ImageClassifier_resnet50, image_path: Path, device: torch.device, tta: bool = True
) -> torch.Tensor:
    """Class probabilities for one image, averaged over the image and its mirror when ``tta``."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    tensor = make_transforms()(load_image(image_path))
    batch = torch.stack([tensor, tensor.flip(-1)]) if tta else tensor.unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(batch.to(device)), dim=1).mean(dim=0)

def load_model(model_path: Path, device: torch.device) -> tuple[ImageClassifier_resnet50, list[str]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at {model_path}. Train one first with -train.")
    checkpoint = torch.load(model_path, map_location=device)
    ARCHITECTURE = "resnet50"
    if checkpoint.get("architecture") not in {ARCHITECTURE}:
        raise ValueError(
            "This model checkpoint was not created with ResNet-50. "
            "Train a new model with -train before classifying images."
        )
    classes = checkpoint["classes"]
    model = ImageClassifier_resnet50(len(classes)).to(device)
    model.load_checkpoint_state(checkpoint["model_state"])
    model.eval()
    return model, classes