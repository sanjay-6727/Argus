# modules/m5_search.py
# ─────────────────────────────────────────
# MODULE 5 — Search Engine
# ─────────────────────────────────────────
# Given a detected object label, pulls real
# knowledge from Wikipedia + optional web search.
# Also builds a CLIP-powered visual similarity
# search over a local image index.
#
# This module is used by m6_hud.py automatically.
# Can also be run standalone for testing.
#
# Run: python modules/m5_search.py
# ─────────────────────────────────────────

import sys
import os
import json
import time
import threading
import hashlib
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import WIKIPEDIA_LANG, MAX_WIKI_SENTENCES


# ── Wikipedia search ─────────────────────

class WikiSearch:
    """
    Fast Wikipedia lookup — given a label, returns a short
    summary and a Wikipedia URL.

    Uses the Wikipedia REST API (no API key needed).
    Results are cached locally so repeated queries are instant.
    """

    CACHE_PATH = os.path.expanduser("~/.cache/ironvision/wiki_cache.json")

    def __init__(self):
        self._cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.CACHE_PATH):
            try:
                with open(self.CACHE_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
        with open(self.CACHE_PATH, "w") as f:
            json.dump(self._cache, f, indent=2)

    def search(self, query, sentences=MAX_WIKI_SENTENCES):
        """
        Search Wikipedia for query.
        Returns { title, summary, url, found: bool }
        """
        key = f"{query.lower().strip()}_{sentences}"
        if key in self._cache:
            return self._cache[key]

        import urllib.request
        import urllib.parse

        # Step 1: Search for best matching article title
        search_url = (
            f"https://en.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={urllib.parse.quote(query)}"
            f"&format=json&srlimit=1"
        )
        try:
            with urllib.request.urlopen(search_url, timeout=5) as r:
                data = json.loads(r.read())
            results = data.get("query", {}).get("search", [])
            if not results:
                result = {"found": False, "query": query, "title": query, "summary": "", "url": ""}
                self._cache[key] = result
                self._save_cache()
                return result

            title = results[0]["title"]

            # Step 2: Get the article extract
            extract_url = (
                f"https://en.wikipedia.org/w/api.php?"
                f"action=query&prop=extracts&exsentences={sentences}"
                f"&exintro=1&explaintext=1&titles={urllib.parse.quote(title)}"
                f"&format=json"
            )
            with urllib.request.urlopen(extract_url, timeout=5) as r:
                data = json.loads(r.read())

            pages  = data.get("query", {}).get("pages", {})
            page   = next(iter(pages.values()))
            extract = page.get("extract", "").strip()

            # Clean up — remove references and footnotes
            extract = re.sub(r'\[\d+\]', '', extract)
            extract = re.sub(r'\s+', ' ', extract).strip()

            result = {
                "found":   True,
                "query":   query,
                "title":   title,
                "summary": extract,
                "url":     f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            }

        except Exception as e:
            result = {
                "found":   False,
                "query":   query,
                "title":   query,
                "summary": f"Search failed: {e}",
                "url":     "",
            }

        self._cache[key] = result
        self._save_cache()
        return result

    def search_async(self, query, callback):
        """Run search in background thread."""
        def _run():
            result = self.search(query)
            callback(result)
        threading.Thread(target=_run, daemon=True).start()


# ── CLIP Visual Search ────────────────────

class CLIPSearchIndex:
    """
    Visual similarity search using CLIP embeddings.

    You can build a local index of reference images (things you
    want to recognise quickly) and this will find the closest match
    to a query frame using cosine similarity.

    Useful for: personal objects, custom products, specific items
    you want the system to recognise without prompting.

    Index is stored at ~/.cache/ironvision/clip_index.json
    """

    INDEX_PATH = os.path.expanduser("~/.cache/ironvision/clip_index.json")

    def __init__(self):
        self.model     = None
        self.processor = None
        self.index     = []   # list of { label, embedding, path }
        self._loaded   = False

    def load(self):
        print("[Search/CLIP] Loading CLIP model ...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True

            self._load_index()
            print(f"[Search/CLIP] Loaded. Index has {len(self.index)} items.")
        except ImportError:
            print("[Search/CLIP] transformers not installed. pip install transformers")
        return self

    def _load_index(self):
        if os.path.exists(self.INDEX_PATH):
            try:
                with open(self.INDEX_PATH) as f:
                    self.index = json.load(f)
            except Exception:
                self.index = []

    def _save_index(self):
        os.makedirs(os.path.dirname(self.INDEX_PATH), exist_ok=True)
        with open(self.INDEX_PATH, "w") as f:
            json.dump(self.index, f)

    def embed_image(self, frame):
        """Get CLIP embedding for an OpenCV BGR frame."""
        import torch
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if hasattr(frame, 'shape') else frame
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def embed_text(self, text):
        """Get CLIP embedding for a text label."""
        import torch
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def add_image(self, frame, label, save=True):
        """Add an image + label to the index."""
        if not self._loaded:
            return
        emb = self.embed_image(frame)
        self.index.append({"label": label, "embedding": emb, "type": "image"})
        if save:
            self._save_index()
        print(f"[Search/CLIP] Added '{label}' to index. ({len(self.index)} total)")

    def find_similar(self, frame, top_k=3, threshold=0.25):
        """
        Find the top-k most similar items in the index to the query frame.
        Returns list of { label, similarity } sorted by similarity.
        """
        import numpy as np

        if not self._loaded or not self.index:
            return []

        query_emb = self.embed_image(frame)
        query_vec = np.array(query_emb)

        results = []
        for item in self.index:
            item_vec = np.array(item["embedding"])
            sim = float(np.dot(query_vec, item_vec))  # both are unit vectors
            if sim >= threshold:
                results.append({"label": item["label"], "similarity": sim})

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    def text_classify(self, frame, labels):
        """
        Zero-shot classify a frame against a list of text labels.
        Returns the best matching label + score.
        CLIP compares image embeddings to text embeddings directly.
        """
        import torch
        import numpy as np
        from PIL import Image

        if not self._loaded:
            return None, 0.0

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            text=labels,
            images=pil,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs   = self.model(**inputs)
            logits    = outputs.logits_per_image[0]
            probs     = logits.softmax(dim=-1).cpu().numpy()

        best_idx = int(np.argmax(probs))
        return labels[best_idx], float(probs[best_idx])


# ── Unified search interface ──────────────

class SearchEngine:
    """
    Combines Wikipedia lookup + CLIP visual search.
    This is the module used by m6_hud.py.
    """

    def __init__(self, use_clip=False):
        self.wiki = WikiSearch()
        self.clip = CLIPSearchIndex() if use_clip else None
        if use_clip and self.clip:
            self.clip.load()

    def lookup(self, label):
        """Wikipedia lookup for a detected label."""
        return self.wiki.search(label)

    def lookup_async(self, label, callback):
        """Async Wikipedia lookup."""
        self.wiki.search_async(label, callback)

    def visual_search(self, frame, top_k=3):
        """CLIP visual similarity search."""
        if self.clip:
            return self.clip.find_similar(frame, top_k=top_k)
        return []


# ── Standalone test ───────────────────────

def run():
    """Test search standalone — try a few queries."""
    import cv2
    engine = SearchEngine(use_clip=False)

    test_queries = [
        "laptop computer",
        "coffee mug",
        "smartphone",
        "bicycle",
        "dog",
    ]

    print("\n=== IronVision Search Engine Test ===\n")
    for q in test_queries:
        print(f"Query: '{q}'")
        result = engine.lookup(q)
        if result["found"]:
            print(f"  Title:   {result['title']}")
            print(f"  Summary: {result['summary'][:200]}...")
            print(f"  URL:     {result['url']}")
        else:
            print(f"  Not found.")
        print()


if __name__ == "__main__":
    run()
