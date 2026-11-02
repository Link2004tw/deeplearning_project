import json

from sklearn.model_selection import train_test_split

def make_split(labels, val_fraction=0.1, seed=42):
    indices = list(range(len(labels)))         
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,                 # 10% goes to validation
        stratify=labels,                        
        random_state=seed,                     
    )
    return train_idx, val_idx

def save_split(train_idx, val_idx, path):
    with open(path, "w") as f:
        json.dump({"train": list(map(int, train_idx)), "val": list(map(int, val_idx))}, f)


def load_split(path):
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"]