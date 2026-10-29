"""FakeLoader and TinyEncoder so evaluation/tuning can be built before real data."""
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from src.models.encoder import BaseEncoder 

class FakeDataset(Dataset):
    """(Image, Label)"""

    def __init__(self, n_samples: int = 1000, n_classes: int = 200, image_size: int = 64, seed: int = 42):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.image_size = image_size
        self.generator = torch.Generator().manual_seed(seed)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int):
        image = torch.randn(3, self.image_size, self.image_size, generator=self.generator)
        label = torch.randint(0, self.n_classes, (1,), generator=self.generator).item()
        return image, label

class FakeLoader: 
    """Dah bitala3 data random ya sa7bi"""

    def __init__(self, batch_size: int = 16, num_classes: int = 200, num_batches: int = 10):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_batches = num_batches
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __len__(self) -> int: 
        return self.num_batches

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        images = torch.randn(self.batch_size, 3, 64, 64)
        labels = torch.randint(0, self.num_classes, (self.batch_size, ))
        self.current_batch +=1
        return images, labels

def make_fake_dataloader(batch_size: int = 16, num_classes: int = 200) -> DataLoader:
    images = torch.randn(200, 3, 64, 64)
    labels = torch.randint(0, num_classes, (200, ))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    

class TinyEncoder(BaseEncoder):
    """Tiny cnn encoder, yarab ostor"""

    def __init__(self, feature_dim: int = 64, in_channels: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),)
        # self.conv1 = torch.nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        # self.relu1 = torch.nn.ReLU()
        # self.conv2 = torch.nn.Conv2d(16, feature_dim, kernel_size=3, stride=1, padding=1)
        # self.relu2 = torch.nn.ReLU()
        # self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.relu1(self.conv1(x))
        # x = self.relu2(self.conv2(x))
        # x = self.pool(x)
        return self.features.flatten(1)

    def cam_target_layer(self):
        return self.features[3]
                
    