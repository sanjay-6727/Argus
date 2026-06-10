# modules/m1_stream.py
# ─────────────────────────────────────────
# MODULE 1 — Phone Camera Stream
# ─────────────────────────────────────────
# Connects to IP Webcam app on your iQOO Z9
# Grabs frames, displays live feed with FPS
#
# Run: python modules/m1_stream.py
# Keys: Q = quit, S = save snapshot
# ─────────────────────────────────────────

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import STREAM_URL, SNAPSHOT_URL, DISPLAY_WIDTH, DISPLAY_HEIGHT, FPS_TARGET
from utils.helpers import resize_frame, draw_fps, draw_status_bar, draw_hud_text, FPSCounter, HUD_COLOR_PRIMARY


class PhoneStream:
    """
    Connects to IP Webcam (Android app) and yields frames.

    IP Webcam streams MJPEG over HTTP — OpenCV reads it
    as if it were a regular video file. Zero latency tricks needed.
    """

    def __init__(self, url=STREAM_URL):
        self.url = url
        self.cap = None

    def connect(self):
        print(f"[Stream] Connecting to {self.url} ...")
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            raise ConnectionError(
                f"[Stream] Could not connect to {self.url}\n"
                "  • Is IP Webcam running on your phone?\n"
                "  • Are both devices on the same WiFi network?\n"
                f"  • Check PHONE_IP in utils/config.py (currently: {self.url})"
            )
        # Buffer size 1 — always grab the latest frame, drop stale ones
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[Stream] Connected!")
        return self

    def read(self):
        """Read one frame. Returns (success, frame)."""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.release()


def run():
    """Standalone stream viewer — run this to test your phone connection."""
    fps_counter = FPSCounter()
    frame_interval = 1.0 / FPS_TARGET
    last_frame_time = 0
    snapshot_count = 0

    try:
        with PhoneStream() as stream:
            print("[Stream] Live feed started. Press Q to quit, S to save snapshot.")
            while True:
                now = time.time()

                ret, frame = stream.read()
                if not ret:
                    print("[Stream] Frame grab failed — retrying...")
                    time.sleep(0.1)
                    continue

                # Throttle to FPS_TARGET
                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                fps_counter.tick()
                frame = resize_frame(frame)

                # ── Overlay ───────────────────────────────
                draw_fps(frame, fps_counter.fps)
                draw_hud_text(frame, "MODULE 1 — STREAM", (10, 22), color=HUD_COLOR_PRIMARY, scale=0.5)
                draw_status_bar(frame, f"Source: {STREAM_URL}  |  {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}  |  Q=quit  S=snapshot")

                cv2.imshow("IronVision — Stream", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    fname = f"snapshot_{snapshot_count:03d}.jpg"
                    cv2.imwrite(fname, frame)
                    snapshot_count += 1
                    print(f"[Stream] Saved {fname}")

    except ConnectionError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[Stream] Stopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
