

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

    <style>
        :root {
            color-scheme: dark;
            --background: #0b1220;
            --surface: #121d2d;
            --surface-inner: #0d1725;
            --border: #29364a;
            --text: #e5edf8;
            --muted: #aebed1;
            --primary: #4385f5;
            --primary-hover: #619bff;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background: var(--background);
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", Arial, sans-serif;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }

        .header {
            width: 100%;
            border-bottom: 1px solid var(--border);
            background: #0e1828;
        }

        .header-inner {
            max-width: 1040px;
            margin: 0 auto;
            padding: 22px 28px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .header-icon {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: #234b85;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #8bbaff;
            flex-shrink: 0;
        }

        h1 {
            font-size: clamp(21px, 3vw, 28px);
            font-weight: 750;
            letter-spacing: -0.7px;
            margin: 0;
        }

        .main {
            width: 100%;
            max-width: 1040px;
            margin: 0 auto;
            padding: 36px 28px 60px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
        }

        .card-heading {
            display: flex;
            align-items: center;
            gap: 11px;
            margin-bottom: 22px;
        }

        .card-heading svg {
            color: #8bbaff;
        }

        .card-heading h2 {
            margin: 0;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: -0.2px;
        }

        .camera-container {
            position: relative;
            width: 100%;
            background: #080e18;
            border: 1px solid #2a394e;
            border-radius: 14px;
            overflow: hidden;
            aspect-ratio: 4 / 3;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #camera {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #080e18;
        }

        .button-group {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 22px;
        }

        button {
            appearance: none;
            border: none;
            border-radius: 10px;
            padding: 13px 22px;
            min-height: 46px;
            font-family: inherit;
            font-size: 14px;
            font-weight: 650;
            cursor: pointer;
            transition: background-color 0.2s ease,
                        box-shadow 0.2s ease,
                        transform 0.2s ease;
        }

        #startCamera {
            background: var(--primary);
            color: #fff;
        }

        #startCamera:hover:not(:disabled) {
            background: var(--primary-hover);
            box-shadow: 0 4px 18px rgba(67, 133, 245, 0.25);
            transform: translateY(-1px);
        }

        #analyze {
            background: #234b85;
            color: #eaf2ff;
            border: 1px solid #3565a6;
        }

        #analyze:hover:not(:disabled) {
            background: #305d9c;
            transform: translateY(-1px);
        }

        button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        button:focus-visible {
            outline: 3px solid #9cc4ff;
            outline-offset: 3px;
        }

        .result-container {
            background: var(--surface-inner);
            border: 1px solid #2b3c53;
            border-radius: 12px;
            padding: 22px;
            min-height: 78px;
            display: flex;
            align-items: center;
        }

        #result {
            margin: 0;
            color: var(--muted);
            font-size: 15px;
            font-weight: 500;
            overflow-wrap: anywhere;
        }

        #capture {
            display: none;
        }

        @media (max-width: 600px) {
            .header-inner {
                padding: 18px 20px;
            }

            .main {
                padding: 22px 15px 40px;
            }

            .card {
                padding: 18px;
                border-radius: 14px;
            }

            .button-group {
                flex-direction: column;
            }

            button {
                width: 100%;
            }

            .camera-container {
                aspect-ratio: 4 / 3;
            }

            .result-container {
                padding: 17px;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            button {
                transition: none;
            }
        }
    </style>
</head>

<body>

    <header class="header">
        <div class="header-inner">
            <div class="header-icon" aria-hidden="true">
                <svg width="25" height="25"
                     viewBox="0 0 24 24"
                     fill="none"
                     stroke="currentColor"
                     stroke-width="1.7"
                     stroke-linecap="round"
                     stroke-linejoin="round">
                    <path d="M9 3h6"/>
                    <path d="M10 3v7l-5 8a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-8V3"/>
                    <path d="M8 16h8"/>
                </svg>
            </div>
            <h1>Lab glass recognition</h1>
        </div>
    </header>

    <main class="main">

        <section class="card">
            <div class="card-heading">
                <svg width="21" height="21"
                     viewBox="0 0 24 24"
                     fill="none"
                     stroke="currentColor"
                     stroke-width="1.8"
                     stroke-linecap="round"
                     stroke-linejoin="round"
                     aria-hidden="true">
                    <rect x="2" y="5" width="15" height="14" rx="2"/>
                    <path d="m17 10 5-3v10l-5-3"/>
                </svg>
                <h2>Camera</h2>
            </div>

            <div class="camera-container">
                <video id="camera"
                       autoplay
                       playsinline
                       width="640"
                       height="480"></video>
            </div>

            <div class="button-group">
                <button id="startCamera">Start camera</button>
                <button id="analyze" disabled>Analyze frame</button>
            </div>
        </section>

        <section class="card">
            <div class="card-heading">
                <svg width="21" height="21"
                     viewBox="0 0 24 24"
                     fill="none"
                     stroke="currentColor"
                     stroke-width="1.8"
                     stroke-linecap="round"
                     stroke-linejoin="round"
                     aria-hidden="true">
                    <path d="M4 5h16"/>
                    <path d="M4 10h16"/>
                    <path d="M4 15h10"/>
                    <path d="M4 20h7"/>
                </svg>
                <h2>Analysis result</h2>
            </div>

            <div class="result-container">
                <p id="result" aria-live="polite">
                    Start the camera to begin.
                </p>
            </div>
        </section>

        <canvas id="capture" hidden></canvas>

    </main>

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

    handler = partial(
        WebRequestHandler,
        model=model,
        classes=classes,
        device=device
    )

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


