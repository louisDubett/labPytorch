from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps


IMAGE_SIZE = 224
EVAL_BACKGROUND = (255, 255, 255)
PAD_FILL = (124, 116, 104)


def get_image_size():
    return IMAGE_SIZE

# --------------------------------------------------------------------------- images

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

def _random_background(size: tuple[int, int], rng) -> Image.Image:
    """Solid, gradient or noisy background so the model cannot rely on a black/transparent one."""
    width, height = size
    first = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    second = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    kind = rng.random()
    if kind < 0.4:
        blend = np.zeros((height, width, 1), dtype=np.float32)
    elif kind < 0.7:
        blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None].repeat(width, 1)
    else:
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        blend = ((xs + ys) / 2.0)[:, :, None]
    pixels = first * (1.0 - blend) + second * blend
    noise_rng = np.random.default_rng(rng.getrandbits(32))
    pixels += noise_rng.normal(0.0, rng.uniform(0.0, 20.0), pixels.shape).astype(np.float32)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, "RGB").convert("RGBA")


def _has_transparency(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )




def _zoom_out(image: Image.Image, rng, min_scale: float = 0.3) -> Image.Image:
    """Shrink a square RGBA image and drop it somewhere on a transparent canvas of the same size.

    Test photos show small glassware on a desk; the cutouts fill the whole frame.
    The object centre stays in the middle half of the canvas so later random crops keep it.
    """
    side = image.width
    new_side = max(1, int(side * rng.uniform(min_scale, 1.0)))
    small = image.resize((new_side, new_side), Image.BILINEAR)
    low = max(0, side // 4 - new_side // 2)
    high = max(low, min(side - new_side, 3 * side // 4 - new_side // 2))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(small, (rng.randint(low, high), rng.randint(low, high)))
    return canvas



def load_image(path: Path, rng=None, zoom_out_probability: float = 0.0) -> Image.Image:
    """Open an image as RGB, honouring EXIF rotation.

    Inference (``rng`` is None): transparent pixels are composited onto white.
    Training/validation: the image is padded to a square, optionally zoomed out and
    composited onto a randomised background so a transparent/black background,
    the aspect ratio and the object scale cannot be used as shortcuts.
    """
    with Image.open(path) as image:
        if image.format == "JPEG":
            # Phone photos are huge; let the decoder downscale cheaply first.
            image.draft("RGB", (get_image_size() * 3, get_image_size() * 3))
        image = ImageOps.exif_transpose(image)
        if rng is None:
            if _has_transparency(image):
                image = image.convert("RGBA")
                background = Image.new("RGBA", image.size, EVAL_BACKGROUND + (255,))
                image = Image.alpha_composite(background, image)
            return image.convert("RGB")
        image = _pad_to_square(image.convert("RGBA"), (0, 0, 0, 0))
        if rng.random() < zoom_out_probability:
            image = _zoom_out(image, rng)
        return Image.alpha_composite(_random_background(image.size, rng), image).convert("RGB")



