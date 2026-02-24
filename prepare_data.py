#!/usr/bin/env python3
import json
import os
import urllib.request
from pathlib import Path

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
DATA_DIR = Path(__file__).parent / "data"


def download_squad(out_path):
    if out_path.exists():
        return
    print("Downloading SQuAD v2 train set...")
    urllib.request.urlretrieve(SQUAD_URL, out_path)


def convert_squad_to_sft(raw_path, sft_path, max_samples=2000):
    with open(raw_path) as f:
        data = json.load(f)

    samples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"].strip()
            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue
                question = qa["question"].strip()
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer = answers[0]["text"].strip()
                if len(context) > 50 and len(question) > 10:
                    query = f"Context: {context} Question: {question}"
                    samples.append({"query": query, "response": answer})
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    with open(sft_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(samples)} samples to {sft_path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = DATA_DIR / "train-v2.0.json"
    sft_path = Path(__file__).parent / "squad_sft.jsonl"
    download_squad(raw_path)
    convert_squad_to_sft(raw_path, sft_path)


if __name__ == "__main__":
    main()
