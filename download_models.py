#!/usr/bin/env python3
import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

MODELS = {
    "rwkv7-g1c-1.5b-20260110-ctx8192.pth": {
        "url": "https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1c-1.5b-20260110-ctx8192.pth",
        "size": "3.06 GB",
    },
    "rwkv7-g1d-1.5b-20260212-ctx8192.pth": {
        "url": "https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1d-1.5b-20260212-ctx8192.pth",
        "size": "3.06 GB",
    },
    "rwkv7-g1d-0.4b-20260210-ctx8192.pth": {
        "url": "https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1d-0.4b-20260210-ctx8192.pth",
        "size": "902 MB",
    },
    "rwkv0b4-emb-curriculum.pth": {
        "url": "https://huggingface.co/howard-hou/EmbeddingRWKV/resolve/main/rwkv0b4-emb-curriculum.pth",
        "size": "924 MB",
    },
}

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 // total_size)
    mb_down = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  {percent}% ({mb_down:.1f}/{mb_total:.1f} MB)")
    sys.stdout.flush()

def download_model(name):
    info = MODELS.get(name)
    if not info:
        print(f"Unknown model: {name}")
        return False

    if info["url"] is None:
        print(f"Model '{name}' is not publicly available.")
        return False

    path = MODEL_DIR / name
    if path.exists():
        print(f"Already exists: {name}")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ({info['size']})...")
    urllib.request.urlretrieve(info["url"], path, download_progress)
    print(f"\n  Saved to {path}")
    return True

def main():
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            download_model(name)
    else:
        print("Available models:")
        for name, info in MODELS.items():
            status = "available" if info["url"] else "(not available)"
            print(f"  {name} - {info['size']} [{status}]")
        print(f"\nUsage: python {sys.argv[0]} <model_name>")

if __name__ == "__main__":
    main()
