import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

import labor_glass_model
from image_transforms import make_transforms


def predict_probabilities(model, image, device: torch.device, tta: bool = True) -> torch.Tensor:
    """Return class probabilities for a single image, with optional mirrored TTA."""
    tensor = make_transforms()(image)
    batch = torch.stack([tensor, tensor.flip(-1)]) if tta else tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(batch.to(device))
    return torch.softmax(logits, dim=1).mean(dim=0)


def classify_frame(model, frame, classes, device, tta: bool = True):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    probabilities = predict_probabilities(model, image, device, tta=tta)
    predicted_index = int(probabilities.argmax().item())
    label = classes[predicted_index]
    confidence = float(probabilities[predicted_index].item())
    return label, confidence


def capture_webcam(model, classes, device, tta: bool = True) -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam. Please verify that a camera is connected.")

    print("Press ESC to exit the webcam view.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to grab a frame from the webcam.")
                break

            label, confidence = classify_frame(model, frame, classes, device, tta=tta)
            text = f"{label}: {confidence:.1%}"
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Webcam analysis", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("Escape hit, closing...")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run live webcam inference with the trained PyTorch model from image_classifier.py."
    )
    parser.add_argument(
        "-model",
        "--model",
        default="model.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable mirrored test-time augmentation.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = labor_glass_model.load_model(model_path, device)
    model.eval()

    capture_webcam(model, classes, device, tta=not args.no_tta)

