# Argus — Modular AR Vision Pipeline

Phone camera → WiFi stream → Laptop AI pipeline → Iron Man HUD

## Modules (build in order)

| # | Module | File | What it does |
|---|--------|------|--------------|
| 1 | Stream | `modules/m1_stream.py` | Connects to IP Webcam, grabs frames, shows live feed |
| 2 | Depth | `modules/m2_depth.py` | Runs MiDaS on each frame, overlays depth map |
| 3 | Detect | `modules/m3_detect.py` | GroundingDINO open-vocab detection, draws bounding boxes |
| 4 | Understand | `modules/m4_understand.py` | Sends frame + detections to LLaVA/GPT-4o, gets description |
| 5 | Search | `modules/m5_search.py` | CLIP embeddings + Wikipedia/web lookup for identified objects |
| 6 | HUD | `modules/m6_hud.py` | Full pipeline combined, Iron Man overlay display |

## Setup

```bash
pip install opencv-python torch torchvision timm requests pillow openai
```

## Phone setup
1. Install **IP Webcam** app (Pavlov Media) from Play Store
2. Open app → scroll to bottom → tap **Start server**
3. Note the URL shown (e.g. `http://192.168.1.5:8080`)
4. Set `PHONE_IP` in `utils/config.py`

## Run each module independently
```bash
python modules/m1_stream.py     # Test stream first
python modules/m2_depth.py      # Add depth
python modules/m3_detect.py     # Add detection
python modules/m4_understand.py # Add VLM brain
python modules/m6_hud.py        # Full pipeline
```
