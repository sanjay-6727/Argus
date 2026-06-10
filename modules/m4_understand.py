# modules/m4_understand.py
# ─────────────────────────────────────────
# MODULE 4 — VLM Understanding (The Brain)
# ─────────────────────────────────────────
# Sends a frame + detections to a Vision-Language
# Model and gets a detailed, JARVIS-style description.
#
# Supports:
#   - GPT-4o via OpenAI API (best quality, needs key)
#   - LLaVA via Ollama (free, local, needs GPU/patience)
#
# Run: python modules/m4_understand.py
# Keys: Q = quit, SPACE = trigger analysis
#       A = auto mode (analyse every N seconds)
# ─────────────────────────────────────────

import cv2
import sys
import os
import time
import threading
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import (
    VLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_URL, OLLAMA_MODEL, VLM_SYSTEM_PROMPT,
    FPS_TARGET, DISPLAY_WIDTH, DISPLAY_HEIGHT
)
from utils.helpers import (
    resize_frame, frame_to_base64, draw_fps, draw_status_bar,
    draw_hud_text, wrap_text, FPSCounter,
    HUD_COLOR_PRIMARY, HUD_COLOR_SECONDARY, HUD_COLOR_TEXT
)
from modules.m1_stream import PhoneStream
from modules.m3_detect import OpenVocabDetector, draw_detections


class VLMBrain:
    """
    Vision-Language Model interface — the understanding core.

    Takes a frame (and optional detection context) and returns
    a rich natural-language description of what it sees.

    Two backends:
    ─ OpenAI GPT-4o: best quality, ~1–2s latency, costs ~$0.01/query
    ─ Ollama LLaVA:  free, local, ~5–20s on CPU, instant on GPU
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False
        self.last_result = None
        self.last_query_time = 0

    def _build_prompt(self, detections=None):
        """Build the user prompt, injecting detection context if available."""
        base = "Look at this image carefully. What do you see?"

        if detections:
            top = detections[:3]  # top 3 most confident detections
            labels = ", ".join(
                f"{d['label']} ({d['confidence']:.0%})"
                for d in top
            )
            base += f"\n\nMy detector found: {labels}. Confirm or correct these, and tell me more."

        return base

    def query_openai(self, frame, detections=None):
        """Query GPT-4o with a base64 frame."""
        import openai

        api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return "[Error] No OpenAI API key. Set OPENAI_API_KEY in config.py or as env var."

        client = openai.OpenAI(api_key=api_key)
        b64 = frame_to_base64(frame, quality=80)

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type":      "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": self._build_prompt(detections)},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[GPT-4o Error] {e}"

    def query_ollama(self, frame, detections=None):
        """Query LLaVA running locally via Ollama."""
        import requests
        import json

        b64 = frame_to_base64(frame, quality=75)

        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": VLM_SYSTEM_PROMPT + "\n\n" + self._build_prompt(detections),
            "images": [b64],
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            return resp.json().get("response", "[No response]").strip()
        except requests.exceptions.ConnectionError:
            return (
                "[Ollama Error] Cannot connect to Ollama.\n"
                "  • Run: ollama serve\n"
                f"  • Then: ollama pull {OLLAMA_MODEL}"
            )
        except Exception as e:
            return f"[Ollama Error] {e}"

    def query(self, frame, detections=None):
        """Route query to correct provider."""
        if VLM_PROVIDER == "openai":
            return self.query_openai(frame, detections)
        else:
            return self.query_ollama(frame, detections)

    def query_async(self, frame, detections=None, callback=None):
        """Run query in background thread. callback(result) called on completion."""
        if self._busy:
            return  # Don't stack queries

        def _run():
            with self._lock:
                self._busy = True
            try:
                result = self.query(frame, detections)
                self.last_result = result
                self.last_query_time = time.time()
                if callback:
                    callback(result)
            finally:
                self._busy = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()


def draw_analysis_panel(frame, result, is_loading=False):
    """
    Draw the JARVIS analysis panel — right side overlay.
    Shows the VLM response text as a semi-transparent panel.
    """
    if not result and not is_loading:
        return

    h, w = frame.shape[:2]
    panel_x = w // 2
    panel_w = w - panel_x - 10
    panel_y = 50
    line_h  = 18
    max_chars = panel_w // 7  # approx chars that fit at our font scale

    lines = []
    if is_loading:
        lines = ["[ ANALYSING... ]", "", "JARVIS is processing"]
    elif result:
        lines.append("[ JARVIS ANALYSIS ]")
        lines.append("")
        wrapped = []
        for para in result.split("\n"):
            wrapped.extend(wrap_text(para.strip(), max_chars=max_chars))
            wrapped.append("")
        lines.extend(wrapped[:16])  # max 16 lines

    # Semi-transparent dark background
    if lines:
        panel_h = len(lines) * line_h + 20
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x - 5, panel_y - 10),
                      (w - 5, panel_y + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(lines):
            y = panel_y + i * line_h + line_h
            color = HUD_COLOR_PRIMARY if (i == 0 or line.startswith("•")) else HUD_COLOR_TEXT
            scale = 0.42
            draw_hud_text(frame, line, (panel_x, y), color=color, scale=scale)


def run():
    """Standalone VLM viewer — stream + optional detection + VLM analysis."""
    detector = OpenVocabDetector()
    detector.load()

    brain = VLMBrain()
    fps_counter = FPSCounter()
    frame_interval = 1.0 / FPS_TARGET
    last_frame_time = 0

    detections  = []
    analysis    = None
    is_loading  = False
    auto_mode   = False
    auto_interval = 5.0   # seconds between auto queries
    last_auto_time = 0
    detect_every   = 8
    frame_count    = 0

    def on_result(text):
        nonlocal analysis, is_loading
        analysis   = text
        is_loading = False
        print(f"\n[JARVIS] {text}\n")

    def trigger_analysis(frame, dets):
        nonlocal is_loading
        if not is_loading:
            is_loading = True
            analysis_frame = frame.copy()
            brain.query_async(analysis_frame, dets, callback=on_result)

    try:
        with PhoneStream() as stream:
            print("[Understand] Running.")
            print("  SPACE = trigger analysis")
            print("  A     = toggle auto mode (every 5s)")
            print("  Q     = quit")
            print(f"  VLM provider: {VLM_PROVIDER}")

            while True:
                now = time.time()
                ret, raw_frame = stream.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                frame = resize_frame(raw_frame)
                frame_count += 1
                fps_counter.tick()

                # Periodic detection
                if frame_count % detect_every == 0:
                    detections = detector.detect(frame)

                display = frame.copy()
                draw_detections(display, detections)
                draw_analysis_panel(display, analysis, is_loading)

                # Auto mode trigger
                if auto_mode and not is_loading:
                    if now - last_auto_time >= auto_interval:
                        last_auto_time = now
                        trigger_analysis(frame, detections)

                draw_fps(display, fps_counter.fps)
                draw_hud_text(display, "MODULE 4 — UNDERSTAND", (10, 22),
                              color=HUD_COLOR_PRIMARY, scale=0.5)

                auto_str = f"AUTO {auto_interval:.0f}s" if auto_mode else "MANUAL"
                status = f"{auto_str}  |  Provider: {VLM_PROVIDER}  |  SPACE=analyse  A=auto  Q=quit"
                draw_status_bar(display, status)

                cv2.imshow("IronVision — Understand", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    trigger_analysis(frame, detections)
                elif key == ord('a'):
                    auto_mode = not auto_mode
                    print(f"[Understand] Auto mode {'ON' if auto_mode else 'OFF'}")

    except ConnectionError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[Understand] Stopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
