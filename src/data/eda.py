import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.captions import (Vocabulary, clean_captions, find_typos,
                               fix_typos, split_description)
from src.data.load_data import get_class_names, load_data
from src.data.splits import make_split, save_split
from src.data.transform import denormalize, get_train_transform


def save_fig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=150)
    plt.close(fig)


# ---------- classes ----------
def class_distribution(labels, class_names, out_dir):
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), np.sort(counts))
    ax.set_xlabel("class (sorted by size)")
    ax.set_ylabel("images")
    ax.set_title("Class distribution (train)")
    save_fig(fig, out_dir, "class_distribution.png")
    return {"classes": len(counts), "min": int(counts.min()),
            "max": int(counts.max()), "mean": float(counts.mean())}


# ---------- images ----------
def scan_images(ds):
    """Decode every image: record size/mode, a thumbnail hash, and corrupt files."""
    records, corrupt = [], []
    for i in range(len(ds)):
        try:
            img = ds[i]["image"]
            img.load()
            thumb = img.convert("L").resize((16, 16)).tobytes()
            records.append({"i": i, "w": img.size[0], "h": img.size[1],
                            "mode": img.mode,
                            "hash": hashlib.md5(thumb).hexdigest()})
        except Exception:
            corrupt.append(i)
    return records, corrupt


def image_stats(records, out_dir):
    w = np.array([r["w"] for r in records])
    h = np.array([r["h"] for r in records])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(w, h, s=3, alpha=0.4)
    axes[0].set_title("Width vs height")
    axes[1].hist(w / h, bins=40)
    axes[1].set_title("Aspect ratio (w/h)")
    axes[2].hist(w * h / 1e3, bins=40)
    axes[2].set_title("Area (thousand pixels)")
    save_fig(fig, out_dir, "image_stats.png")
    return {"width": [int(w.min()), int(w.mean()), int(w.max())],
            "height": [int(h.min()), int(h.mean()), int(h.max())],
            "modes": dict(Counter(r["mode"] for r in records))}


def find_duplicates(records):
    groups = {}
    for r in records:
        groups.setdefault(r["hash"], []).append(r["i"])
    return [g for g in groups.values() if len(g) > 1]


# ---------- captions ----------
def caption_stats(descriptions, out_dir):
    raw = [split_description(d) for d in descriptions]
    cleaned = [clean_captions(c) for c in raw]
    flat = [c for caps in cleaned for c in caps]

    typo_map = find_typos(flat)
    flat_fixed = [fix_typos(c, typo_map) for c in flat]

    lengths = [len(c.split()) for c in flat_fixed]
    words = Counter(w for c in flat_fixed for w in c.split())
    raw_vocab = {w for caps in raw for c in caps for w in c.lower().split()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(lengths, bins=range(0, max(lengths) + 2))
    axes[0].set_title("Caption length (words)")
    top = words.most_common(20)
    axes[1].barh([w for w, _ in top][::-1], [n for _, n in top][::-1])
    axes[1].set_title("Top 20 words")
    save_fig(fig, out_dir, "caption_stats.png")

    return {
        "images_without_10_captions": sum(len(c) != 10 for c in raw),
        "captions_removed_as_duplicate_or_empty":
            sum(len(r) for r in raw) - sum(len(c) for c in cleaned),
        "length_min_mean_max": [min(lengths), float(np.mean(lengths)), max(lengths)],
        "vocab_raw": len(raw_vocab),
        "vocab_cleaned": len(words),
        "words_seen_once": sum(n == 1 for n in words.values()),
        "typos_fixed": len(typo_map),
    }, typo_map


# ---------- augmentation ----------
def augmentation_examples(ds, out_dir, image_size=128, n_images=4, n_aug=5, aug=None):
    tf = get_train_transform(image_size, aug)
    fig, axes = plt.subplots(n_images, n_aug + 1, figsize=(2.5 * (n_aug + 1), 2.5 * n_images))
    for r in range(n_images):
        img = ds[r * 100]["image"].convert("RGB")
        axes[r, 0].imshow(img.resize((image_size, image_size)))
        axes[r, 0].set_title("original" if r == 0 else "")
        for c in range(1, n_aug + 1):
            aug_img = denormalize(tf(img)).permute(1, 2, 0).numpy()
            axes[r, c].imshow(aug_img)
            axes[r, c].set_title(f"aug {c}" if r == 0 else "")
    for ax in axes.ravel():
        ax.axis("off")
    save_fig(fig, out_dir, "augmentation_examples.png")


# ---------- run everything ----------
def run_eda(cache_dir="data/cub", out_dir="outputs/figures", data_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    train, test = load_data(cache_dir)
    names = get_class_names(train)
    labels = list(train["label"])
    summary = {}

    summary["classes"] = class_distribution(labels, names, out_dir)

    tr_records, tr_corrupt = scan_images(train)
    te_records, te_corrupt = scan_images(test)
    summary["images"] = image_stats(tr_records, out_dir)
    summary["corrupt"] = {"train": tr_corrupt, "test": te_corrupt}
    summary["duplicates_in_train"] = len(find_duplicates(tr_records))
    shared = {r["hash"] for r in tr_records} & {r["hash"] for r in te_records}
    summary["train_test_overlap"] = len(shared)

    summary["captions"], typo_map = caption_stats(train["description"], out_dir)

    train_idx, val_idx = make_split(labels)
    save_split(train_idx, val_idx, Path(data_dir) / "split.json")
    vocab_caps = [fix_typos(c, typo_map)
                  for i in train_idx
                  for c in clean_captions(split_description(train[i]["description"]))]
    vocab = Vocabulary(min_freq=3)
    vocab.build(vocab_caps)
    vocab.save(Path(data_dir) / "vocab.json")
    summary["vocab_size_min_freq_3"] = len(vocab)

    augmentation_examples(train, out_dir)

    with open(Path(data_dir) / "typo_map.json", "w") as f:
        json.dump(typo_map, f, indent=2)
    with open(Path(data_dir) / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="data/cub")
    p.add_argument("--out_dir", default="outputs/figures")
    args = p.parse_args()
    run_eda(args.cache_dir, args.out_dir)