import torchvision.transforms as transforms

from dataset import (
  PlanktonDataset,
  filter_dataset,
  get_class_counts,
  get_class_to_idx
)

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])