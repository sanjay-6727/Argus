# utils/helpers.py
# ─────────────────────────────────────────
# Shared helpers used across all modules
# ─────────────────────────────────────────

import cv2
import numpy as np
import time
import base64
from PIL import Image
import io
from utils.config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    HUD_COLOR_PRIMARY, HUD_COLOR_SECONDARY, HUD_COLOR_DANGER, HUD_COLOR_TEXT,
    HUD_FONT_SCALE, HUD_THICKNESS
)

# ── Frame utilities ───────────────────────

def resize_frame(frame, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT):
    """Resize frame to target display size."""
    return cv2.resize(frame, (width, height))


def frame_to_pil(frame):
    """Convert OpenCV BGR frame to PIL RGB Image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_frame(pil_img):
    """Convert PIL RGB Image to OpenCV BGR frame."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def frame_to_base64(frame, quality=85):
    """Encode OpenCV frame to base64 JPEG string (for API calls)."""
    pil_img = frame_to_pil(frame)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── HUD drawing utilities ─────────────────

def draw_hud_text(frame, text, pos, color=HUD_COLOR_TEXT, scale=HUD_FONT_SCALE, thickness=HUD_THICKNESS):
    """Draw text with a subtle dark shadow for readability on any background."""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Shadow
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hud_box(frame, x1, y1, x2, y2, label="", confidence=None, color=HUD_COLOR_PRIMARY):
    """Draw a detection bounding box with corner brackets (Iron Man style)."""
    thickness = 2
    corner_len = 14  # length of corner bracket lines

    # Corner brackets instead of full box — cleaner HUD look
    corners = [
        ((x1, y1), (x1 + corner_len, y1), (x1, y1 + corner_len)),   # top-left
        ((x2, y1), (x2 - corner_len, y1), (x2, y1 + corner_len)),   # top-right
        ((x1, y2), (x1 + corner_len, y2), (x1, y2 - corner_len)),   # bottom-left
        ((x2, y2), (x2 - corner_len, y2), (x2, y2 - corner_len)),   # bottom-right
    ]
    for corner, h_pt, v_pt in corners:
        cv2.line(frame, corner, h_pt, color, thickness, cv2.LINE_AA)
        cv2.line(frame, corner, v_pt, color, thickness, cv2.LINE_AA)

    # Label with confidence
    if label:
        conf_str = f" {confidence:.0%}" if confidence is not None else ""
        tag = f"{label}{conf_str}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, HUD_FONT_SCALE, HUD_THICKNESS)
        # Dark pill background behind label
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 0, 0), -1)
        draw_hud_text(frame, tag, (x1 + 4, y1 - 4), color=color)


def draw_fps(frame, fps):
    """Draw FPS counter top-right."""
    draw_hud_text(frame, f"{fps:.0f} fps", (DISPLAY_WIDTH - 70, 22), color=HUD_COLOR_PRIMARY, scale=0.45)


def draw_status_bar(frame, text, color=HUD_COLOR_PRIMARY):
    """Draw a one-line status bar at the bottom of the frame."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (0, 0, 0), -1)
    draw_hud_text(frame, text, (10, h - 8), color=color, scale=0.45)


# ── FPS tracker ───────────────────────────

class FPSCounter:
    def __init__(self, smoothing=10):
        self._times = []
        self._smoothing = smoothing

    def tick(self):
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._smoothing:
            self._times.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── Misc ──────────────────────────────────

def wrap_text(text, max_chars=55):
    """Wrap long text into lines for HUD display."""
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line += (" " if line else "") + word
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
