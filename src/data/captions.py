import re
import json
from collections import Counter
from difflib import get_close_matches

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

def split_description(description: str):
    """
    Split the description into a list of sentences.
    """
    return [caption.strip() for caption in description.splitlines() if caption.strip()]

def clean_caption(caption):

    caption= caption.lower().replace("'", "")
    caption= re.sub(r'[^a-z0-9 ]', " ", caption)
    return caption.strip()


def clean_captions(captions):
    seen, result = set(), []
    for caption in captions:
        caption = clean_caption(caption)
        if caption and caption not in seen:
            seen.add(caption)
            result.append(caption)
    return result


def find_typos(captions, rare_max=2, common_min=20, cutoff=0.85):
    """
    Typo = a rare word that looks almost identical to a common word.
    Returns {typo: correction}. REVIEW this map by eye before using it.
    """
    counts = Counter(w for c in captions for w in c.split())
    common = [w for w, n in counts.items() if n >= common_min]
    typo_map = {}
    for word, n in counts.items():
        if n <= rare_max and len(word) >= 5 and not word.isdigit():
            match = get_close_matches(word, common, n=1, cutoff=cutoff)
            if match:
                typo_map[word] = match[0]
    return typo_map


def fix_typos(caption: str, typo_map: dict):
    return " ".join(typo_map.get(w, w) for w in caption.split())


class Vocabulary:
    """Word <-> id mapping. Build it from TRAIN captions only (no leakage)."""

    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.itos = [PAD, SOS, EOS, UNK]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def build(self, captions):
        counts = Counter(w for c in captions for w in c.split())
        for word, n in counts.most_common():
            if n >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def encode(self, caption):
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in caption.split()]
        return [self.stoi[SOS]] + ids + [self.stoi[EOS]]

    def decode(self, ids):
        words = [self.itos[i] for i in ids]
        return " ".join(w for w in words if w not in (PAD, SOS, EOS))

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path) as f:
            vocab.itos = json.load(f)
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        return vocab