from torchvision import transforms
import torch

DEFAULT_AUG = {
    "crop_min_scale": 0.6,  
    "flip_prob": 0.5,       
    "rotation_deg": 15,
    "color_jitter": 0.2,   
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_train_transform(image_size=128,aug=None):
    aug = {**DEFAULT_AUG, **(aug or {})}
    steps = [transforms.RandomResizedCrop(image_size, scale=(aug["crop_min_scale"], 1.0))]
    if aug["flip_prob"] > 0:
        steps.append(transforms.RandomHorizontalFlip(aug["flip_prob"]))
    if aug["rotation_deg"] > 0:
        steps.append(transforms.RandomRotation(aug["rotation_deg"]))
    if aug["color_jitter"] > 0:
        j = aug["color_jitter"]
        steps.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j))
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def get_test_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def denormalize(tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)