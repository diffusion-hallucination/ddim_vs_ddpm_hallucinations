from torchvision import datasets, transforms
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
# from gaussian_data import *

# Loading Image Dataset (from the dataset path root_dir)
class ImageDataset(Dataset):
    def __init__(self, root_dir, image_size=64, channels=3):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.channels = channels
        
        # Get all PNG files
        self.image_paths = list(self.root_dir.glob("**/*.png"))
        print(f"Found {len(self.image_paths)} PNG images")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels)  # Scale to [-1, 1]
        ])
    
    def __len__(self):
        # len.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # getitem.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        img_path = self.image_paths[idx]
        
        # Load image
        if self.channels == 1:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        return image