import torchvision
from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL = (124, 116, 104)

def get_image_size():
    return IMAGE_SIZE

def _pad_to_square(image: Image.Image, fill) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas

class PadToSquare:
    """Letterbox to a square so tall phone photos are not squashed."""

    def __init__(self, fill: tuple[int, int, int] = PAD_FILL) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return _pad_to_square(image, self.fill)

def make_transforms(training: bool = False) -> transforms.Compose:
    if training:
        image_transforms = [
            PadToSquare(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20, fill=PAD_FILL)], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3, fill=PAD_FILL),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
        ]
    else:
        image_transforms = [
            PadToSquare(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(image_transforms)