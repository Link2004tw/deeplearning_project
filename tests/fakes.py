"""FakeLoader and TinyEncoder so evaluation/tuning can be built before real data."""
import torch 
from torch.utils.data import DataLoader, Dataset

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

    def __init__(self, n_samples: int = 1000, batch_size: int = 32, n_classes: int = 200, image_size: int = 64, seed: int = 42, shuffle: bool = True):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.image_size = image_size
        self.shuffle = shuffle
        self.dataset = FakeDataset(n_samples, n_classes, image_size, seed)

    def __iter__(self):
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
            generator=torch.Generator().manuel_seed(
                self.dataset.generator.initial_seed() if hasattr(self.dataset.generator, 'initial_seed') else 42
                ))
        for images, labels in loader:
            yield images.float(), labels.long()

    def __len__(self) -> int: 
        return (self.n_samples + self.batch_size -1) // self.batch_size

class TinyEncoder(BaseEncoder):
    """Tiny cnn encoder, yarab ostor"""

    def __init__(self, feature_dim: int = 63, in_channels: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = torch.nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLu()
        self.conv2 = torch.nn.Conv2d(16, feature_dim, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLu()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        return torch.flatten(x, 1)

    def cam_target_layer(self):
        return self.conv2
                
    