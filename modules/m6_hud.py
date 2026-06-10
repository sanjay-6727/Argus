# modules/m6_hud.py
# ─────────────────────────────────────────
# MODULE 6 — Full Iron Man HUD Pipeline
# ─────────────────────────────────────────
# Combines all modules into a single
# real-time AR vision system.
#
#  Stream → Depth → Detection → VLM → Search
#                                     ↓
#                               Iron Man HUD
#
# Run: python modules/m6_hud.py
#
# Keys:
#   Q     — quit
#   SPACE — trigger VLM analysis on current frame
#   D     — toggle depth overlay
#   A     — toggle auto-analysis mode
#   F     — freeze / unfreeze frame
#   S     — save screenshot
#   +/-   — adjust depth overlay opacity
# ─────────────────────────────────────────

import cv2
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import (
    FPS_TARGET, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    DEPTH_ALPHA, HUD_COLOR_PRIMARY, HUD_COLOR_SECONDARY,
    HUD_COLOR_TEXT, DETECT_PROMPT, VLM_PROVIDER
)
from utils.helpers import (
    resize_frame, draw_fps, draw_status_bar,
    draw_hud_text, draw_hud_box, wrap_text, FPSCounter,
    HUD_COLOR_DANGER
)
from modules.m1_stream import PhoneStream
from modules.m2_depth  import DepthEstimator
from modules.m3_detect import OpenVocabDetector, draw_detections, confidence_color
from modules.m4_understand import VLMBrain, draw_analysis_panel
from modules.m5_search import SearchEngine


# ── Pipeline state ────────────────────────

class PipelineState:
    def __init__(self):
        self.frame           = None
        self.depth_map       = None
        self.detections      = []
        self.analysis        = None
        self.wiki_results    = {}   # label → wiki result
        self.is_analysing    = False
        self.show_depth      = True
        self.auto_mode       = False
        self.frozen          = False
        self.depth_alpha     = DEPTH_ALPHA
        self.frame_count     = 0
        self.screenshot_count = 0
        self.active_label    = None   # currently selected detection for wiki


# ── HUD rendering ─────────────────────────

def render_hud(frame, state: PipelineState, fps, depth_estimator):
    """
    Render the full Iron Man HUD onto the frame.
    All drawing happens here — keeps each module clean.
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # ── Depth overlay ──────────────────────
    if state.show_depth and state.depth_map is not None:
        import cv2 as cv
        import numpy as np

        d = state.depth_map
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = numpy.zeros_like(d, dtype=np.uint8)

        depth_vis = cv.applyColorMap(norm, cv.COLORMAP_INFERNO)
        cv.addWeighted(display, 1 - state.depth_alpha, depth_vis, state.depth_alpha, 0, display)

    # ── Bounding boxes ──────────────────────
    for det in state.detections:
        x1, y1, x2, y2 = det["box"]
        color = confidence_color(det["confidence"])

        # Depth label for this detection
        depth_label = ""
        if state.depth_map is not None:
            depth_label = depth_estimator.estimate_distance_label(state.depth_map, (x1, y1, x2, y2))

        label = f"{det['label']} [{depth_label}]" if depth_label else det["label"]
        draw_hud_box(display, x1, y1, x2, y2,
                     label=label,
                     confidence=det["confidence"],
                     color=color)

        # Wiki snippet below the box if available
        wiki = state.wiki_results.get(det["label"])
        if wiki and wiki.get("found") and wiki.get("summary"):
            snippet = wiki["summary"][:60] + "..."
            draw_hud_text(display, snippet, (x1, y2 + 14),
                          color=HUD_COLOR_SECONDARY, scale=0.35)

    # ── VLM analysis panel ──────────────────
    draw_analysis_panel(display, state.analysis, state.is_analysing)

    # ── Corner HUD elements ─────────────────

    # Top-left: title + object count
    draw_hud_text(display, "IRONVISION v1.0", (10, 22),
                  color=HUD_COLOR_PRIMARY, scale=0.55)
    draw_hud_text(display, f"{len(state.detections)} object(s) detected",
                  (10, 44), color=HUD_COLOR_SECONDARY, scale=0.42)

    # Top-right: FPS + mode indicators
    draw_fps(display, fps)
    modes = []
    if state.show_depth:  modes.append("DEPTH")
    if state.auto_mode:   modes.append("AUTO")
    if state.frozen:      modes.append("FROZEN")
    if state.is_analysing: modes.append("ANALYSING...")
    if modes:
        draw_hud_text(display, "  ".join(modes),
                      (w - 180, 44), color=HUD_COLOR_SECONDARY, scale=0.38)

    # ── Scanning animation (top of frame) ──
    scan_x = int((time.time() * 120) % w)
    cv2.line(display, (scan_x, 0), (scan_x, 4), HUD_COLOR_PRIMARY, 1, cv2.LINE_AA)

    # ── Corner brackets (overall frame) ────
    corner_size = 18
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (cx, cy), (dx, dy) in zip(corners, dirs):
        cv2.line(display, (cx, cy), (cx + dx * corner_size, cy), HUD_COLOR_PRIMARY, 1)
        cv2.line(display, (cx, cy), (cx, cy + dy * corner_size), HUD_COLOR_PRIMARY, 1)

    # ── Status bar ──────────────────────────
    provider = VLM_PROVIDER.upper()
    draw_status_bar(display,
        f"SPACE=analyse  D=depth  A=auto  F=freeze  S=save  Q=quit  | VLM:{provider}")

    return display


# ── Main pipeline ─────────────────────────

def run():
    # Load all models
    print("=" * 50)
    print("  IronVision — Full Pipeline")
    print("=" * 50)

    depth_est  = DepthEstimator()
    detector   = OpenVocabDetector()
    brain      = VLMBrain()
    search_eng = SearchEngine(use_clip=False)

    depth_est.load()
    detector.load()
    # VLM + search are API/lazy — no explicit load needed

    state = PipelineState()
    fps_counter = FPSCounter()
    frame_interval = 1.0 / FPS_TARGET
    last_frame_time = 0
    last_auto_time  = 0
    auto_interval   = 6.0   # seconds

    # Stagger heavy processing
    depth_every  = 3
    detect_every = 6

    def on_analysis(text):
        state.analysis    = text
        state.is_analysing = False

    def trigger_analysis():
        if not state.is_analysing and state.frame is not None:
            state.is_analysing = True
            state.analysis = None
            brain.query_async(state.frame.copy(), state.detections, callback=on_analysis)

    def on_wiki(label, result):
        state.wiki_results[label] = result

    print("\n[IronVision] All systems online.")
    print(f"  Depth model : {depth_est.model_type}")
    print(f"  VLM backend : {VLM_PROVIDER}")
    print(f"  Detect prompt: {DETECT_PROMPT[:60]}")
    print("\nPress SPACE to analyse, Q to quit.\n")

    try:
        with PhoneStream() as stream:
            while True:
                now = time.time()
                ret, raw_frame = stream.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                state.frame_count += 1
                fps_counter.tick()

                if not state.frozen:
                    frame = resize_frame(raw_frame)
                    state.frame = frame

                    # Depth estimation
                    if state.frame_count % depth_every == 0:
                        state.depth_map = depth_est.estimate(frame)

                    # Detection
                    if state.frame_count % detect_every == 0:
                        new_dets = detector.detect(frame)
                        state.detections = new_dets

                        # Kick off wiki search for new top detections
                        for det in new_dets[:2]:
                            label = det["label"]
                            if label not in state.wiki_results:
                                search_eng.lookup_async(
                                    label,
                                    callback=lambda r, l=label: on_wiki(l, r)
                                )

                # Auto VLM
                if state.auto_mode and not state.is_analysing:
                    if now - last_auto_time >= auto_interval:
                        last_auto_time = now
                        trigger_analysis()

                # Render
                display = render_hud(
                    state.frame if state.frame is not None else resize_frame(raw_frame),
                    state, fps_counter.fps, depth_est
                )

                cv2.imshow("IronVision", display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    trigger_analysis()
                elif key == ord('d'):
                    state.show_depth = not state.show_depth
                elif key == ord('a'):
                    state.auto_mode = not state.auto_mode
                    print(f"[HUD] Auto mode {'ON' if state.auto_mode else 'OFF'}")
                elif key == ord('f'):
                    state.frozen = not state.frozen
                    print(f"[HUD] {'Frozen' if state.frozen else 'Unfrozen'}")
                elif key == ord('s'):
                    fname = f"ironvision_{state.screenshot_count:04d}.jpg"
                    cv2.imwrite(fname, display)
                    state.screenshot_count += 1
                    print(f"[HUD] Screenshot saved: {fname}")
                elif key == ord('+') or key == ord('='):
                    state.depth_alpha = min(1.0, state.depth_alpha + 0.05)
                elif key == ord('-'):
                    state.depth_alpha = max(0.0, state.depth_alpha - 0.05)

    except ConnectionError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[IronVision] Shutting down.")
    finally:
        cv2.destroyAllWindows()
        print("[IronVision] Done.")


if __name__ == "__main__":
    run()
