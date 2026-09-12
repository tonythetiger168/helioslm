"""Image preprocessing for vision encoder."""
import torch
import torch.nn.functional as F
from typing import Union, List
from PIL import Image


class ImageProcessor:
    """Preprocess images for ViT input."""

    def __init__(self, img_size: int = 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def preprocess(self, image: Union[Image.Image, str, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image: PIL Image, file path, or tensor [C, H, W]

        Returns:
            tensor: [1, 3, 224, 224]
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            # Resize and convert to tensor
            image = image.resize((self.img_size, self.img_size))
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        else:
            tensor = image.float()

        # Normalize
        tensor = (tensor - self.mean) / self.std

        return tensor.unsqueeze(0)

    def preprocess_batch(self, images: List[Union[Image.Image, str]]) -> torch.Tensor:
        """Preprocess a batch of images."""
        tensors = [self.preprocess(img) for img in images]
        return torch.cat(tensors, dim=0)


import numpy as np
