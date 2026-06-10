# modules/m3_detect.py
# ─────────────────────────────────────────
# MODULE 3 — Open-Vocabulary Object Detection
# ─────────────────────────────────────────
# Uses GroundingDINO — detects ANY object
# you describe in plain English. No fixed classes.
#
# Run: python modules/m3_detect.py
# Keys: Q = quit, E = edit detection prompt
#       SPACE = freeze/unfreeze frame
# ─────────────────────────────────────────

import cv2
import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import (
    DETECT_PROMPT, DETECT_BOX_THRESHOLD, DETECT_TEXT_THRESHOLD,
    FPS_TARGET, DISPLAY_WIDTH, DISPLAY_HEIGHT
)
from utils.helpers import (
    resize_frame, draw_fps, draw_status_bar,
    draw_hud_text, draw_hud_box, FPSCounter,
    HUD_COLOR_PRIMARY, HUD_COLOR_SECONDARY, HUD_COLOR_DANGER
)
from modules.m1_stream import PhoneStream


# Confidence → colour mapping
def confidence_color(conf):
    if conf >= 0.7:
        return HUD_COLOR_PRIMARY    # Green-cyan: high confidence
    elif conf >= 0.45:
        return HUD_COLOR_SECONDARY  # Amber: medium
    else:
        return HUD_COLOR_DANGER     # Red: low confidence


class OpenVocabDetector:
    """
    GroundingDINO: open-vocabulary object detector.

    Unlike YOLO (fixed 80 classes), GroundingDINO takes a
    text prompt and finds whatever you describe — "a red cup",
    "person wearing glasses", "vintage camera", anything.

    How it works:
    - Vision backbone encodes image features
    - BERT-based text encoder processes your prompt
    - Cross-attention fuses visual + text features
    - Decoder predicts bounding boxes for text-matched regions

    First run: downloads ~700MB of model weights.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self._gdino = None

    def load(self):
        print("[Detect] Loading GroundingDINO ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Detect] Using device: {self.device}")

        # Install groundingdino if not present
        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T
        except ImportError:
            print("[Detect] GroundingDINO not installed. Installing...")
            os.system("pip install groundingdino-py -q")
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T

        self._gdino_predict = predict
        self._T = T

        # Download weights if needed
        weights_path = os.path.expanduser("~/.cache/ironvision/groundingdino_swint_ogc.pth")
        config_path  = os.path.expanduser("~/.cache/ironvision/GroundingDINO_SwinT_OGC.py")

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        if not os.path.exists(weights_path):
            print("[Detect] Downloading GroundingDINO weights (~700MB) ...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                weights_path
            )

        if not os.path.exists(config_path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                config_path
            )

        self.model = load_model(config_path, weights_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[Detect] GroundingDINO loaded.")
        return self

    def detect(self, frame, prompt=DETECT_PROMPT,
               box_threshold=DETECT_BOX_THRESHOLD,
               text_threshold=DETECT_TEXT_THRESHOLD):
        """
        Run detection on a BGR frame with a text prompt.

        Returns list of dicts:
          { label, confidence, box: (x1,y1,x2,y2) in pixel coords }
        """
        from groundingdino.util.inference import predict
        from PIL import Image as PILImage
        import groundingdino.datasets.transforms as T

        h, w = frame.shape[:2]
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(pil_img, None)
        image_tensor = image_tensor.to(self.device)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        results = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            # box is cx,cy,w,h normalised — convert to pixel x1y1x2y2
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            results.append({
                "label":      phrase.strip(),
                "confidence": float(logit),
                "box":        (x1, y1, x2, y2),
                "box_norm":   box.tolist(),  # keep normalised for later modules
            })

        # Sort by confidence descending
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results


def draw_detections(frame, detections):
    """Draw all detections onto the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = confidence_color(det["confidence"])
        draw_hud_box(frame, x1, y1, x2, y2,
                     label=det["label"],
                     confidence=det["confidence"],
                     color=color)

    # Detection count top-left
    draw_hud_text(frame, f"{len(detections)} object(s)", (10, 48),
                  color=HUD_COLOR_PRIMARY, scale=0.45)


def run():
    """Standalone detection viewer."""
    detector = OpenVocabDetector()
    detector.load()

    fps_counter   = FPSCounter()
    frame_interval = 1.0 / FPS_TARGET
    last_frame_time = 0

    current_prompt = DETECT_PROMPT
    detections     = []
    process_every  = 5    # Detection is heavier — run every N frames
    frame_count    = 0
    frozen         = False
    frozen_frame   = None

    try:
        with PhoneStream() as stream:
            print(f"[Detect] Running. Prompt: '{current_prompt}'")
            print("[Detect] Keys: Q=quit, SPACE=freeze, E=new prompt")

            while True:
                now = time.time()
                ret, raw_frame = stream.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                frame_count += 1
                fps_counter.tick()

                if frozen and frozen_frame is not None:
                    frame = frozen_frame.copy()
                else:
                    frame = resize_frame(raw_frame)
                    if not frozen:
                        frozen_frame = frame.copy()

                # Run detection periodically
                if not frozen and frame_count % process_every == 0:
                    detections = detector.detect(frame, prompt=current_prompt)

                display = frame.copy()
                draw_detections(display, detections)
                draw_fps(display, fps_counter.fps)
                draw_hud_text(display, "MODULE 3 — DETECT", (10, 22),
                              color=HUD_COLOR_PRIMARY, scale=0.5)

                freeze_label = "FROZEN" if frozen else f"Prompt: {current_prompt[:40]}"
                draw_status_bar(display,
                    f"{freeze_label}  |  SPACE=freeze  E=prompt  Q=quit")

                cv2.imshow("IronVision — Detect", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    frozen = not frozen
                    print(f"[Detect] {'Frozen' if frozen else 'Unfrozen'}")
                elif key == ord('e'):
                    cv2.destroyAllWindows()
                    new_prompt = input("[Detect] Enter new detection prompt: ").strip()
                    if new_prompt:
                        current_prompt = new_prompt
                        print(f"[Detect] Prompt updated to: '{current_prompt}'")
                    detections = []

    except ConnectionError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[Detect] Stopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
