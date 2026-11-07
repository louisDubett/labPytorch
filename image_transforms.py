

from torchvision import transforms

from image_tool import get_image_size, PadToSquare, PAD_FILL

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(training: bool = False) -> transforms.Compose:
    if training:
        image_transforms = [
            PadToSquare(),
            transforms.RandomResizedCrop(get_image_size(), scale=(0.6, 1.0), ratio=(0.8, 1.25)),
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
            transforms.Resize((get_image_size(), get_image_size())),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(image_transforms)