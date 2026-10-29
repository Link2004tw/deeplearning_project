from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
          self.forward_handle.remove()
          self.backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device 

        if image_tensor.dim() == 3: 
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device=device)
        image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)

        if class_idx is None: 
            class_idx = logits.argmax(dim=1).item()
        
        self.model.zero_grad()

        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        activations =self.activations[0]
        gradients =self.gradients[0]
        weights = gradients.mean(dim=(1, 2))

        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cam.cpu().numpy()
        h, w = image_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h))

        return cam


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
                
        
def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
):

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay  

def denormalize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
)-> np.ndarray: 
    image = image.copy()
    if image.ndim == 3 and image.shape[0] in (3, 1):
        image = image.transpose(1, 2, 0)

    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image