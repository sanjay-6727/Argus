# modules/m2_depth.py
# ─────────────────────────────────────────
# MODULE 2 — Monocular Depth Estimation
# ─────────────────────────────────────────
# Uses MiDaS to estimate depth from a single
# RGB frame. No depth sensor needed.
#
# Run: python modules/m2_depth.py
# Keys: Q = quit, D = toggle depth overlay,
#       1/2/3 = switch colormap
# ─────────────────────────────────────────

import cv2
import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import DEPTH_MODEL, DEPTH_ALPHA, FPS_TARGET
from utils.helpers import (
    resize_frame, frame_to_pil, draw_fps,
    draw_status_bar, draw_hud_text, FPSCounter, HUD_COLOR_PRIMARY
)
from modules.m1_stream import PhoneStream


# Colormaps to cycle through — each gives a different visual feel
COLORMAPS = {
    '1': (cv2.COLORMAP_INFERNO,  "Inferno"),   # default — warm/cool
    '2': (cv2.COLORMAP_PLASMA,   "Plasma"),    # purple-yellow
    '3': (cv2.COLORMAP_TURBO,    "Turbo"),     # full spectrum
}


class DepthEstimator:
    """
    Wraps MiDaS v3.1 for per-frame depth estimation.

    How it works:
    - MiDaS is a transformer trained to predict relative depth
      from a single RGB image.
    - Output is a 2D map where each pixel's value represents
      its estimated distance (relative, not metric).
    - DPT_Large is most accurate but slowest (~200ms on CPU).
      Use MiDaS_small for real-time on slower machines.
    """

    def __init__(self, model_type=DEPTH_MODEL):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = None

    def load(self):
        print(f"[Depth] Loading MiDaS model: {self.model_type} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Depth] Using device: {self.device}")

        # Load from torch hub (downloads ~500MB on first run)
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.model_type,
            trust_repo=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Matching transform for chosen model
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        print("[Depth] Model loaded.")
        return self

    @torch.no_grad()
    def estimate(self, frame):
        """
        Run depth estimation on a BGR OpenCV frame.
        Returns a depth map as a numpy float32 array (same HxW as input).
        Values are relative depth — higher = farther.
        """
        # MiDaS transform expects a numpy RGB array, not PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).to(self.device)

        prediction = self.model(input_tensor)

        # Interpolate back to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy()

    def to_colormap(self, depth_map, colormap_id=cv2.COLORMAP_INFERNO):
        """Convert float depth map to a colourised BGR image for display."""
        # Normalise to 0–255
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            normalized = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_map, dtype=np.uint8)

        return cv2.applyColorMap(normalized, colormap_id)

    def estimate_distance_label(self, depth_map, bbox=None):
        """
        Returns a rough qualitative distance label for a region.
        If bbox (x1,y1,x2,y2) given, uses that region. Else uses center crop.
        Depth is relative so we bucket into Near/Mid/Far.
        """
        h, w = depth_map.shape
        if bbox:
            x1, y1, x2, y2 = bbox
            region = depth_map[y1:y2, x1:x2]
        else:
            cx, cy = w // 2, h // 2
            region = depth_map[cy-30:cy+30, cx-30:cx+30]

        if region.size == 0:
            return "??"

        val = region.mean()
        # Normalise against full frame range
        full_range = depth_map.max() - depth_map.min()
        if full_range == 0:
            return "??"

        ratio = (val - depth_map.min()) / full_range  # 0=closest, 1=farthest

        if ratio < 0.25:
            return "NEAR"
        elif ratio < 0.55:
            return "MID"
        else:
            return "FAR"


def run():
    """Standalone depth viewer — stream + depth overlay."""
    estimator = DepthEstimator()
    estimator.load()

    fps_counter = FPSCounter()
    frame_interval = 1.0 / FPS_TARGET
    last_frame_time = 0

    show_depth   = True
    current_cmap = '1'
    depth_map    = None
    process_every = 3   # Run depth every N frames to keep UI smooth
    frame_count  = 0

    try:
        with PhoneStream() as stream:
            print("[Depth] Running. Keys: Q=quit, D=toggle depth, 1/2/3=colormap")
            while True:
                now = time.time()
                ret, frame = stream.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                frame = resize_frame(frame)
                frame_count += 1
                fps_counter.tick()

                # Run depth every few frames — it's slow on CPU
                if frame_count % process_every == 0:
                    depth_map = estimator.estimate(frame)

                display = frame.copy()

                # Blend depth overlay
                if show_depth and depth_map is not None:
                    cmap_id, cmap_name = COLORMAPS[current_cmap]
                    depth_vis = estimator.to_colormap(depth_map, cmap_id)

                    # Distance label at center
                    label = estimator.estimate_distance_label(depth_map)
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, h // 2

                    # Blend original + depth
                    display = cv2.addWeighted(frame, 1 - DEPTH_ALPHA, depth_vis, DEPTH_ALPHA, 0)

                    # Center crosshair + distance
                    cv2.drawMarker(display, (cx, cy), HUD_COLOR_PRIMARY,
                                   cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
                    draw_hud_text(display, label, (cx + 14, cy + 5),
                                  color=HUD_COLOR_PRIMARY, scale=0.6)
                    draw_hud_text(display, f"[{cmap_name}]", (w - 90, h - 36),
                                  color=HUD_COLOR_PRIMARY, scale=0.4)

                draw_fps(display, fps_counter.fps)
                draw_hud_text(display, "MODULE 2 — DEPTH", (10, 22),
                              color=HUD_COLOR_PRIMARY, scale=0.5)
                mode_str = "DEPTH ON" if show_depth else "DEPTH OFF"
                draw_status_bar(display,
                    f"{mode_str}  |  Model: {DEPTH_MODEL}  |  D=toggle  1/2/3=map  Q=quit")

                cv2.imshow("IronVision — Depth", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_depth = not show_depth
                elif key in (ord('1'), ord('2'), ord('3')):
                    current_cmap = chr(key)

    except ConnectionError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[Depth] Stopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
