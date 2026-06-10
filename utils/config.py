# utils/config.py
# ─────────────────────────────────────────
# Central config — edit this file only
# ─────────────────────────────────────────

# ── Phone stream ──────────────────────────
PHONE_IP   = "192.168.1.6"   # Change to your phone's IP shown in IP Webcam app
PHONE_PORT = 8080
STREAM_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"
SNAPSHOT_URL = f"http://{PHONE_IP}:{PHONE_PORT}/shot.jpg"

# ── Display ───────────────────────────────
DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540
FPS_TARGET     = 15          # Limit processing FPS to avoid overloading laptop

# ── Depth (MiDaS) ─────────────────────────
DEPTH_MODEL    = "DPT_Large"  # Options: "DPT_Large" (best), "DPT_Hybrid", "MiDaS_small" (fastest)
DEPTH_ALPHA    = 0.6          # Overlay opacity: 0.0 = no overlay, 1.0 = full depth map

# ── Detection (GroundingDINO) ─────────────
# Default prompt — what to look for. Can be anything in natural language.
DETECT_PROMPT  = "object . person . text . product . animal . food . vehicle"
DETECT_BOX_THRESHOLD  = 0.35
DETECT_TEXT_THRESHOLD = 0.25

# ── VLM (Understanding) ───────────────────
VLM_PROVIDER   = "local"    # "openai" (GPT-4o) or "local" (LLaVA via ollama)
OPENAI_API_KEY = ""          # Paste your key here, or set env var OPENAI_API_KEY
OPENAI_MODEL   = "gpt-4o"
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_MODEL = "moondream"  # fast, fits your GPU

# VLM system prompt — defines the "Iron Man" personality
VLM_SYSTEM_PROMPT = """You are JARVIS, an advanced visual AI assistant built into smart glasses.
When given an image, identify the most prominent object or scene and provide:
1. What it is (be specific — brand, model, species, etc. if visible)
2. Key facts (3-4 bullet points max)
3. Distance estimate if relevant
4. Any actionable insight (price range, safety info, interesting fact)
Be concise. Max 5 sentences total. Think like a heads-up display, not a textbook."""

# ── Search ────────────────────────────────
WIKIPEDIA_LANG = "en"
MAX_WIKI_SENTENCES = 3

# ── HUD colours (BGR for OpenCV) ──────────
HUD_COLOR_PRIMARY   = (0, 255, 180)   # Cyan-green — Iron Man HUD feel
HUD_COLOR_SECONDARY = (0, 200, 255)   # Amber
HUD_COLOR_DANGER    = (0, 60, 255)    # Red
HUD_COLOR_TEXT      = (255, 255, 255) # White
HUD_FONT_SCALE      = 0.55
HUD_THICKNESS       = 1
