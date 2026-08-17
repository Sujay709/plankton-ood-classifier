from io import text_encoding
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
import torchvision.models as models

from dataset import get_class_counts, filter_dataset, get_class_to_idx, split_dataset, PlanktonDataset, test_indices, train_indices

DATA_DIR = Path("data/raw")

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_counts = get_class_counts(DATA_DIR)
known_classes = filter_dataset(class_counts)
class_to_idx = get_class_to_idx(known_classes)

if __name__ == '__main__':
  dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=class_to_idx, transform=transform)
  train_indices, test_indices = split_dataset(dataset)
  
  test_dataset = Subset(dataset, test_indices)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)