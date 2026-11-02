import torch
from torch.utils.data import DataLoader, Dataset

from src.data.load_data import get_class_names, load_data
from src.data.splits import make_split
from src.data.transform import get_test_transform, get_train_transform


class CUBDataset(Dataset):
    """Wraps the HuggingFace dataset; returns (image_tensor, label)."""

    def __init__(self, hf_dataset, indices, transform=None):
        self.data = hf_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.data[self.indices[i]]
        image = row["image"].convert("RGB")   
        if self.transform:
            image = self.transform(image)
        return image, row["label"]


def build_dataloaders(cfg: dict):
    """
    cfg = the `data:` section of the YAML. Keys (all optional):
      root, image_size, batch_size, num_workers, val_split, seed, subset, augmentation
    Returns: train_loader, val_loader, test_loader, meta
    meta = {"num_classes": int, "class_names": list[str]}
    """
    train_data, test_data = load_data(cfg.get("root", "data/cub"))
    size = cfg.get("image_size", 128)

    labels = list(train_data["label"])        
    train_idx, val_idx = make_split(labels, cfg.get("val_split", 0.1), cfg.get("seed", 42))

    if cfg.get("subset"):
        train_idx = train_idx[: cfg["subset"]]

    train_tf = get_train_transform(size, cfg.get("augmentation"))
    eval_tf = get_test_transform(size)

    train_ds = CUBDataset(train_data, train_idx, train_tf)
    val_ds = CUBDataset(train_data, val_idx, eval_tf)      
    test_ds = CUBDataset(test_data, range(len(test_data)), eval_tf)

    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 2)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=nw, pin_memory=pin)

    class_names = get_class_names(train_data)
    meta = {"num_classes": len(class_names), "class_names": class_names}
    return train_loader, val_loader, test_loader, meta