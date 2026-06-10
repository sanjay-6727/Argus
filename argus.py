# ironvision.py
# ═══════════════════════════════════════════════════════════
#  IRONVISION — Unified Real-Time Vision Pipeline
#  Stream → Depth → Detection → VLM Description → Search
# ═══════════════════════════════════════════════════════════
#
#  Setup:
#    pip install torch torchvision timm opencv-python pillow
#                requests transformers numpy groundingdino-py
#    ollama pull moondream   (or llava:7b)
#
#  Phone: Install "IP Webcam" app → Start server → note IP
#  Edit PHONE_IP below, then: python ironvision.py
#
#  Keys:  SPACE = describe what camera sees
#         D     = toggle depth overlay
#         A     = auto-describe mode (every 5s)
#         F     = freeze frame
#         S     = save screenshot
#         E     = change detection prompt
#         Q     = quit
# ═══════════════════════════════════════════════════════════

import cv2
import torch
import numpy as np
import sys
import os
import time
import threading
import json
import re
import urllib.request
import urllib.parse
import base64
import requests
from PIL import Image as PILImage
from io import BytesIO

# ──────────────────────────────────────────
#  CONFIG  — edit these
# ──────────────────────────────────────────

PHONE_IP        = "192.168.1.6"          # Just the IP, no http://
PHONE_PORT      = 8080
STREAM_URL      = f"http://{PHONE_IP}:{PHONE_PORT}/video"

DISPLAY_WIDTH   = 960
DISPLAY_HEIGHT  = 540

FPS_TARGET      = 30

# Depth
DEPTH_MODEL     = "MiDaS_small"          # Fastest. Use DPT_Large for accuracy (slow)
DEPTH_ALPHA     = 0.35                   # Blend strength of depth overlay

# Detection
DETECT_PROMPT           = "person . bottle . cup . phone . laptop . chair . table . book . bag . keyboard . mouse . monitor . food . plant"
DETECT_BOX_THRESHOLD    = 0.30
DETECT_TEXT_THRESHOLD   = 0.25
DETECT_EVERY_N_FRAMES   = 8              # Run detection every N frames

# Depth runs every N frames (depth is lighter than detection)
DEPTH_EVERY_N_FRAMES    = 3

# VLM — local Ollama
OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_MODEL    = "moondream"            # or "llava" if you pulled that
AUTO_INTERVAL   = 5.0                    # seconds between auto-describe calls

# Wikipedia
WIKI_SENTENCES  = 2

# ──────────────────────────────────────────
#  HUD COLOURS  (BGR)
# ──────────────────────────────────────────

C_PRIMARY   = (0,   255, 180)   # Cyan-green
C_SECONDARY = (0,   200, 255)   # Amber-ish
C_DANGER    = (40,   40, 255)   # Red
C_TEXT      = (220, 220, 220)   # Off-white
C_DIM       = (100, 100, 100)

FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────

class FPSCounter:
    def __init__(self, window=30):
        self._times = []
        self._window = window
        self.fps = 0.0

    def tick(self):
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) >= 2:
            self.fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])


def resize_frame(frame):
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)


def put(img, text, pos, color=C_TEXT, scale=0.45, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,     thickness,     cv2.LINE_AA)


def draw_fps(img, fps):
    put(img, f"{fps:.1f} FPS", (DISPLAY_WIDTH - 80, 18), C_PRIMARY, 0.45)


def draw_status_bar(img, text):
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 22), (img.shape[1], h), (10, 10, 10), -1)
    put(img, text, (6, h - 6), C_DIM, 0.36)


def conf_color(c):
    if c >= 0.7:  return C_PRIMARY
    if c >= 0.45: return C_SECONDARY
    return C_DANGER


def draw_box(img, x1, y1, x2, y2, label="", confidence=None, color=None):
    color = color or C_PRIMARY
    s = 12   # corner bracket size
    pts = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (px,py),(dx,dy) in zip(pts,dirs):
        cv2.line(img,(px,py),(px+dx*s,py),color,1,cv2.LINE_AA)
        cv2.line(img,(px,py),(px,py+dy*s),color,1,cv2.LINE_AA)

    if label:
        conf_str = f" {confidence:.2f}" if confidence is not None else ""
        full = f"{label}{conf_str}"
        (tw, th), _ = cv2.getTextSize(full, FONT, 0.38, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), (10,10,10), -1)
        put(img, full, (x1 + 3, y1 - 4), color, 0.38)


def wrap_text(text, max_chars=55):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur.strip())
            cur = w + " "
        else:
            cur += w + " "
    if cur.strip():
        lines.append(cur.strip())
    return lines


def frame_to_b64(frame):
    """Convert BGR frame to base64 JPEG string for Ollama."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


# ──────────────────────────────────────────
#  MODULE 1 — PHONE STREAM
# ──────────────────────────────────────────

class PhoneStream:
    def __init__(self):
        self.cap = None

    def connect(self):
        print(f"[Stream] Connecting to {STREAM_URL} ...")
        self.cap = cv2.VideoCapture(STREAM_URL)
        if not self.cap.isOpened():
            raise ConnectionError(
                f"[Stream] Could not connect to {STREAM_URL}\n"
                "  • Is IP Webcam running?\n"
                "  • Same WiFi network?\n"
                f"  • Check PHONE_IP at top of ironvision.py"
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[Stream] Connected!")
        return self

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):  return self.connect()
    def __exit__(self, *_): self.release()


# ──────────────────────────────────────────
#  MODULE 2 — DEPTH ESTIMATION (MiDaS)
# ──────────────────────────────────────────

class DepthEstimator:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    def load(self):
        print(f"[Depth] Loading MiDaS ({DEPTH_MODEL}) ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL, trust_repo=True)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if DEPTH_MODEL in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        print(f"[Depth] Loaded on {self.device}")
        return self

    @torch.no_grad()
    def estimate(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic", align_corners=False
        ).squeeze()
        return pred.cpu().numpy()

    def to_colormap(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(depth_map, dtype=np.uint8)
        return cv2.applyColorMap(norm, cmap)

    def distance_label(self, depth_map, bbox=None):
        h, w = depth_map.shape
        if bbox:
            x1, y1, x2, y2 = bbox
            region = depth_map[y1:y2, x1:x2]
        else:
            cx, cy = w//2, h//2
            region = depth_map[cy-30:cy+30, cx-30:cx+30]
        if region.size == 0:
            return "?"
        val = region.mean()
        full_range = depth_map.max() - depth_map.min()
        if full_range == 0:
            return "?"
        ratio = (val - depth_map.min()) / full_range
        if ratio < 0.25:  return "NEAR"
        if ratio < 0.55:  return "MID"
        return "FAR"


# ──────────────────────────────────────────
#  MODULE 3 — OPEN-VOCAB DETECTION (GroundingDINO)
# ──────────────────────────────────────────

class Detector:
    def __init__(self):
        self.model = None
        self.device = None

    def load(self):
        print("[Detect] Loading GroundingDINO ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from groundingdino.util.inference import load_model
        except ImportError:
            print("[Detect] Installing groundingdino-py ...")
            os.system("pip install groundingdino-py -q")
            from groundingdino.util.inference import load_model

        weights = os.path.expanduser("~/.cache/ironvision/groundingdino_swint_ogc.pth")
        config  = os.path.expanduser("~/.cache/ironvision/GroundingDINO_SwinT_OGC.py")
        os.makedirs(os.path.dirname(weights), exist_ok=True)

        if not os.path.exists(weights):
            print("[Detect] Downloading weights (~700MB) ...")
            urllib.request.urlretrieve(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                weights
            )
        if not os.path.exists(config):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                config
            )

        self.model = load_model(config, weights).to(self.device)
        self.model.eval()
        print("[Detect] Loaded.")
        return self

    def detect(self, frame, prompt=DETECT_PROMPT):
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        h, w = frame.shape[:2]
        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        img_t, _ = transform(pil, None)
        img_t = img_t.to(self.device)

        boxes, logits, phrases = predict(
            model=self.model, image=img_t, caption=prompt,
            box_threshold=DETECT_BOX_THRESHOLD,
            text_threshold=DETECT_TEXT_THRESHOLD,
        )

        results = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            x1 = max(0, int((cx - bw/2) * w))
            y1 = max(0, int((cy - bh/2) * h))
            x2 = min(w, int((cx + bw/2) * w))
            y2 = min(h, int((cy + bh/2) * h))
            results.append({
                "label":      phrase.strip(),
                "confidence": float(logit),
                "box":        (x1, y1, x2, y2),
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results


# ──────────────────────────────────────────
#  MODULE 4 — VLM BRAIN (Ollama local)
# ──────────────────────────────────────────

class VLMBrain:
    """Sends frame + detected labels to Ollama (moondream/llava) for description."""

    def _build_prompt(self, detections):
        if detections:
            labels = ", ".join(d["label"] for d in detections[:5])
            return (
                f"You are a smart vision assistant. I can see these objects: {labels}. "
                "Briefly describe the scene in 2-3 sentences. "
                "For the most prominent object, tell me: what it is, what it's used for, "
                "and any interesting fact about it. Be direct and informative."
            )
        return (
            "You are a smart vision assistant. Look at this image carefully. "
            "Describe what you see in 2-3 sentences. Identify the main objects, "
            "their purpose, and anything interesting about the scene."
        )

    def query(self, frame, detections=None):
        """
        Send frame to Ollama, return description string.
        Crops to the most prominent detection if available — faster + more accurate.
        """
        # If we have a high-confidence detection, crop to it for sharper VLM focus
        query_frame = frame
        if detections:
            best = detections[0]
            x1, y1, x2, y2 = best["box"]
            # Only crop if the box is large enough to be meaningful
            bw, bh = x2 - x1, y2 - y1
            if bw > 80 and bh > 80:
                # Add padding
                pad = 20
                h, w = frame.shape[:2]
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                query_frame = frame[cy1:cy2, cx1:cx2]
        query_frame = cv2.resize(query_frame,(320, 320),interpolation=cv2.INTER_AREA)
        b64 = frame_to_b64(query_frame)
        prompt = self._build_prompt(detections)

        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,   # Lower = more factual, less hallucination
                        "num_predict": 120,   # Cap tokens for speed
                    }
                },
                timeout=120
            )
            data = resp.json()
            return data.get("response", "No response").strip()
        except requests.exceptions.ConnectionError:
            return "Ollama not running. Start it with: ollama serve"
        except Exception as e:
            return f"VLM error: {e}"

    def query_async(self, frame, detections, callback):
        """Non-blocking — runs in background thread, calls callback(text) when done."""
        def _run():
            result = self.query(frame, detections)
            callback(result)
        threading.Thread(target=_run, daemon=True).start()


# ──────────────────────────────────────────
#  MODULE 5 — WIKIPEDIA SEARCH
# ──────────────────────────────────────────

class WikiSearch:
    CACHE = os.path.expanduser("~/.cache/ironvision/wiki_cache.json")

    def __init__(self):
        self._cache = {}
        if os.path.exists(self.CACHE):
            try:
                with open(self.CACHE) as f:
                    self._cache = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.CACHE), exist_ok=True)
        with open(self.CACHE, "w") as f:
            json.dump(self._cache, f)

    def search(self, query):
        key = query.lower().strip()
        if key in self._cache:
            return self._cache[key]

        try:
            # Find title
            url = (f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                   f"&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1")
            with urllib.request.urlopen(url, timeout=4) as r:
                data = json.loads(r.read())
            hits = data.get("query", {}).get("search", [])
            if not hits:
                result = {"found": False, "summary": ""}
                self._cache[key] = result
                return result

            title = hits[0]["title"]
            # Get extract
            url2 = (f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                    f"&exsentences={WIKI_SENTENCES}&exintro=1&explaintext=1"
                    f"&titles={urllib.parse.quote(title)}&format=json")
            with urllib.request.urlopen(url2, timeout=4) as r:
                data = json.loads(r.read())
            page = next(iter(data["query"]["pages"].values()))
            summary = re.sub(r'\[\d+\]', '', page.get("extract", "")).strip()
            summary = re.sub(r'\s+', ' ', summary)

            result = {"found": True, "title": title, "summary": summary}
        except Exception:
            result = {"found": False, "summary": ""}

        self._cache[key] = result
        self._save()
        return result

    def search_async(self, query, callback):
        threading.Thread(target=lambda: callback(self.search(query)), daemon=True).start()


# ──────────────────────────────────────────
#  PIPELINE STATE
# ──────────────────────────────────────────

class State:
    def __init__(self):
        self.frame          = None
        self.depth_map      = None
        self.detections     = []
        self.description    = None      # Latest VLM text
        self.is_describing  = False
        self.wiki           = {}        # label → wiki result
        self.show_depth     = True
        self.auto_mode      = False
        self.frozen         = False
        self.depth_alpha    = DEPTH_ALPHA
        self.frame_count    = 0
        self.screenshots    = 0
        self.detect_prompt  = DETECT_PROMPT
        # For change-detection — only re-trigger VLM when scene changes
        self._last_labels   = []


# ──────────────────────────────────────────
#  HUD RENDERER
# ──────────────────────────────────────────

def render(frame, state: State, fps: float, depth_est: DepthEstimator):
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Depth overlay ────────────────
    if state.show_depth and state.depth_map is not None:
        dvis = depth_est.to_colormap(state.depth_map)
        cv2.addWeighted(out, 1 - state.depth_alpha, dvis, state.depth_alpha, 0, out)

    # ── Detection boxes ──────────────
    for det in state.detections:
        x1, y1, x2, y2 = det["box"]
        color = conf_color(det["confidence"])

        depth_lbl = ""
        if state.depth_map is not None:
            depth_lbl = depth_est.distance_label(state.depth_map, (x1, y1, x2, y2))

        label = f"{det['label']} [{depth_lbl}]" if depth_lbl else det["label"]
        draw_box(out, x1, y1, x2, y2, label=label,
                 confidence=det["confidence"], color=color)

        # Wiki snippet under box
        wiki = state.wiki.get(det["label"])
        if wiki and wiki.get("found") and wiki.get("summary"):
            snip = wiki["summary"][:65] + ("..." if len(wiki["summary"]) > 65 else "")
            put(out, snip, (x1, y2 + 14), C_SECONDARY, 0.33)

    # ── VLM Description panel (right side) ──
    panel_x = w - 310
    if state.description or state.is_describing:
        overlay = out.copy()
        cv2.rectangle(overlay, (panel_x - 8, 30), (w - 4, h - 30), (10,10,10), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

        put(out, "▸ VISION ANALYSIS", (panel_x, 48), C_PRIMARY, 0.42)
        cv2.line(out, (panel_x, 54), (w - 6, 54), C_PRIMARY, 1)

        if state.is_describing:
            # Animated dots
            dots = "." * (int(time.time() * 2) % 4)
            put(out, f"Analysing{dots}", (panel_x, 75), C_SECONDARY, 0.4)
        elif state.description:
            lines = wrap_text(state.description, max_chars=42)
            for i, line in enumerate(lines[:12]):
                put(out, line, (panel_x, 72 + i * 18), C_TEXT, 0.37)

    # ── Top-left info ──────────────────
    put(out, "IRONVISION", (10, 22), C_PRIMARY, 0.55)
    put(out, f"{len(state.detections)} object(s)", (10, 42), C_SECONDARY, 0.4)

    # ── Top-right FPS + modes ─────────
    draw_fps(out, fps)
    modes = []
    if state.show_depth:   modes.append("DEPTH")
    if state.auto_mode:    modes.append("AUTO")
    if state.frozen:       modes.append("FROZEN")
    if modes:
        put(out, "  ".join(modes), (w - 180, 38), C_SECONDARY, 0.36)

    # ── Scanning line ─────────────────
    sx = int((time.time() * 130) % w)
    cv2.line(out, (sx, 0), (sx, 3), C_PRIMARY, 1, cv2.LINE_AA)

    # ── Corner brackets ───────────────
    cs = 16
    for (px,py),(dx,dy) in zip([(0,0),(w-1,0),(0,h-1),(w-1,h-1)],
                                [(1,1),(-1,1),(1,-1),(-1,-1)]):
        cv2.line(out,(px,py),(px+dx*cs,py),C_PRIMARY,1,cv2.LINE_AA)
        cv2.line(out,(px,py),(px,py+dy*cs),C_PRIMARY,1,cv2.LINE_AA)

    # ── Status bar ────────────────────
    draw_status_bar(out,
        f"SPACE=describe  D=depth  A=auto({AUTO_INTERVAL}s)  F=freeze  E=prompt  S=save  Q=quit"
        f"  |  model:{OLLAMA_MODEL}")

    return out


# ──────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────

def main():
    print("=" * 56)
    print("  IRONVISION — Real-Time Vision Pipeline")
    print("=" * 56)
    print(f"  Phone stream : {STREAM_URL}")
    print(f"  Depth model  : {DEPTH_MODEL}")
    print(f"  VLM model    : {OLLAMA_MODEL} (via Ollama)")
    print(f"  Detect prompt: {DETECT_PROMPT[:50]}...")
    print("=" * 56)

    # Load models
    depth_est = DepthEstimator().load()
    detector  = Detector().load()
    brain     = VLMBrain()
    wiki      = WikiSearch()

    state = State()
    fps_counter = FPSCounter()
    last_frame_t  = 0
    last_auto_t   = 0
    frame_interval = 1.0 / FPS_TARGET

    def on_description(text):
        state.description   = text
        state.is_describing = False

    def trigger_describe():
        if not state.is_describing and state.frame is not None:
            state.is_describing = True
            state.description   = None
            brain.query_async(state.frame.copy(), state.detections, on_description)

    def on_wiki(label, result):
        state.wiki[label] = result

    print("\n[IronVision] All systems online. Press SPACE to describe, Q to quit.\n")

    try:
        with PhoneStream() as stream:
            while True:
                now = time.time()
                ret, raw = stream.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                # Throttle display to FPS_TARGET
                if now - last_frame_t < frame_interval:
                    continue
                last_frame_t = now

                state.frame_count += 1
                fps_counter.tick()

                if not state.frozen:
                    frame = resize_frame(raw)
                    state.frame = frame

                    # ── Depth (lightweight, run often) ───
                    if state.frame_count % DEPTH_EVERY_N_FRAMES == 0:
                        state.depth_map = depth_est.estimate(frame)

                    # ── Detection (heavier, run less often) ─
                    if state.frame_count % DETECT_EVERY_N_FRAMES == 0:
                        dets = detector.detect(frame, prompt=state.detect_prompt)
                        state.detections = dets

                        # Async wiki lookup for new top detections
                        for det in dets[:3]:
                            lbl = det["label"]
                            if lbl not in state.wiki:
                                wiki.search_async(lbl, lambda r, l=lbl: on_wiki(l, r))

                        # Auto-describe if labels changed significantly
                        new_labels = [d["label"] for d in dets[:3]]
                        if state.auto_mode and new_labels != state._last_labels:
                            state._last_labels = new_labels
                            if not state.is_describing:
                                trigger_describe()

                # ── Auto-describe on timer ────────────
                if state.auto_mode and not state.is_describing:
                    if now - last_auto_t >= AUTO_INTERVAL:
                        last_auto_t = now
                        trigger_describe()

                # ── Render HUD ───────────────────────
                base = state.frame if state.frame is not None else resize_frame(raw)
                display = render(base, state, fps_counter.fps, depth_est)
                cv2.imshow("IronVision", display)

                # ── Key handling ─────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    trigger_describe()
                elif key == ord('d'):
                    state.show_depth = not state.show_depth
                elif key == ord('a'):
                    state.auto_mode = not state.auto_mode
                    last_auto_t = now
                    print(f"[IronVision] Auto mode {'ON' if state.auto_mode else 'OFF'}")
                elif key == ord('f'):
                    state.frozen = not state.frozen
                    print(f"[IronVision] {'Frozen' if state.frozen else 'Unfrozen'}")
                elif key == ord('s'):
                    fname = f"ironvision_{state.screenshots:04d}.jpg"
                    cv2.imwrite(fname, display)
                    state.screenshots += 1
                    print(f"[IronVision] Saved {fname}")
                elif key == ord('e'):
                    cv2.destroyAllWindows()
                    p = input("[IronVision] New detection prompt: ").strip()
                    if p:
                        state.detect_prompt = p
                        state.detections = []
                        print(f"[IronVision] Prompt → '{p}'")
                elif key == ord('+') or key == ord('='):
                    state.depth_alpha = min(1.0, state.depth_alpha + 0.05)
                elif key == ord('-'):
                    state.depth_alpha = max(0.0, state.depth_alpha - 0.05)

    except ConnectionError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[IronVision] Stopping...")
    finally:
        cv2.destroyAllWindows()
        print("[IronVision] Done.")


if __name__ == "__main__":
    main()
