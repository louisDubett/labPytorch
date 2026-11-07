import argparse
from typing import Any, Union

import cv2
from pathlib import Path
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
from image_transforms import get_image_size

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.transforms import v2



def make_transforms() -> transforms.Compose:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    image_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(image_transforms)

def predict_probabilities(
        model: ImageClassifier_resnet50, image_frame, device: torch.device, tta: bool = True
) -> torch.Tensor:
    """Class probabilities for one image, averaged over the image and its mirror when ``tta``."""

    transfomer=make_transforms()
    tensor = transfomer(image_frame)
    batch = torch.stack([tensor, tensor.flip(-1)]) if tta else tensor.unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(batch.to(device)), dim=1).mean(dim=0)

def classify_image(img_counter,image_frame, device,model,classes) -> None:
    h, w = image_frame.shape[:2]
    # desired height is get_image_size()
    fx = get_image_size() / h
    fy = fx
    # Scale down by x%
    smallerImage = cv2.resize(image_frame, None, fx=fx, fy=fy)

    img_name = "opencv_frame_{}.png".format(img_counter)
    image_file = img_name
    cv2.imwrite(image_file, smallerImage)

    probabilities = predict_probabilities(model, smallerImage, device)
    predicted_index = int(probabilities.argmax().item())
    print(
        f"Prediction: {classes[predicted_index]} "
        f"(confidence: {probabilities[predicted_index].item():.1%})"
    )

def capture(device,model,classes):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("capture img")
    img_counter =0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        classify_image(img_counter,image_frame=frame, device=device, model=model, classes=classes)
        img_counter += 1

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()

cv2.destroyAllWindows()

if __name__ == '__main__':
    print("press ESC to exit and space to take image")
    parser = argparse.ArgumentParser(
        description="Evaluate a pytorch model with an XML config and produce a test_result XML file."
    )
    parser.add_argument(
        "-model",
        default="model.pth",
        help="Path to pytorch model file.",
    )

    args = parser.parse_args()
    model_path=Path(args.model)

    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes = labor_glass_model.load_model(model_path, device)
    capture(device,model,classes)

