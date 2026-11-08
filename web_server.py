import json
import logging
import socketserver
from functools import partial
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import torch

import labor_glass_model
from analyse_image import classify_frame


HTML_TEXT = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Lab glass recognition</title>
</head>
<body>
    <h1>Lab glass recognition</h1>
    <video id="camera" autoplay playsinline width="640" height="480"></video>
    <br>
    <button id="startCamera">Start camera</button>
    <button id="analyze" disabled>Analyze frame</button>
    <p id="result" aria-live="polite">Start the camera to begin.</p>
    <canvas id="capture" hidden></canvas>
    <script>
        const video = document.getElementById("camera");
        const startButton = document.getElementById("startCamera");
        const analyzeButton = document.getElementById("analyze");
        const result = document.getElementById("result");
        const canvas = document.getElementById("capture");
        let cameraStream;

        startButton.addEventListener("click", async () => {
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({video: true});
                video.srcObject = cameraStream;
                await video.play();
                analyzeButton.disabled = false;
                startButton.disabled = true;
                result.textContent = "Camera ready. Capture a frame to analyze it.";
            } catch (error) {
                result.textContent = `Could not start the camera: ${error.message}`;
            }
        });

        analyzeButton.addEventListener("click", async () => {
            analyzeButton.disabled = true;
            result.textContent = "Analyzing...";
            try {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext("2d").drawImage(video, 0, 0);
                const image = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
                if (!image) {
                    throw new Error("Could not capture a camera frame.");
                }

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {"Content-Type": "image/jpeg"},
                    body: image
                });
                const prediction = await response.json();
                if (!response.ok) {
                    throw new Error(prediction.error || "The server could not analyze the image.");
                }
                result.textContent = `${prediction.label}: ${(prediction.confidence * 100).toFixed(1)}%`;
            } catch (error) {
                result.textContent = `Analysis failed: ${error.message}`;
            } finally {
                analyzeButton.disabled = false;
            }
        });

        window.addEventListener("beforeunload", () => {
            cameraStream?.getTracks().forEach(track => track.stop());
        });
    </script>
</body>
</html>
"""


class WebRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, model, classes, device, **kwargs):
        self.model = model
        self.classes = classes
        self.device = device
        super().__init__(*args, **kwargs)

    def _send_json(self, status_code, payload):
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        html = HTML_TEXT.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "Not found."})
            return

        if self.headers.get_content_type() != "image/jpeg":
            self._send_json(415, {"error": "Send a JPEG image."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length."})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "The image is empty."})
            return

        image_bytes = self.rfile.read(content_length)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            self._send_json(400, {"error": "Could not decode the JPEG image."})
            return

        label, confidence = classify_frame(
            self.model,
            frame,
            self.classes,
            self.device,
        )
        self._send_json(200, {"label": label, "confidence": confidence})


def run(port=8080, model_path=Path("model.pth")):
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = labor_glass_model.load_model(Path(model_path), device)
    model.eval()

    handler = partial(WebRequestHandler, model=model, classes=classes, device=device)
    with socketserver.TCPServer(("", port), handler) as my_server:
        logging.info("Starting httpd on port %s...", port)
        try:
            my_server.serve_forever()
        except KeyboardInterrupt:
            logging.info("Stopping httpd...")


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    elif len(argv) == 3:
        run(port=int(argv[1]), model_path=Path(argv[2]))
    else:
        run()
