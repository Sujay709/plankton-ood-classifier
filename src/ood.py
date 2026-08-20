from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset

from dataset import get_class_to_idx, PlanktonDataset, OOD_CLASSES
from dataset import get_class_counts, filter_dataset, get_class_to_idx, split_dataset, PlanktonDataset, OOD_CLASSES

DATA_DIR = Path("data/raw")

ood_class_to_idx = get_class_to_idx(sorted(OOD_CLASSES))

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_msp_confidences(model, loader, device):
  """
  Run every image in loader through model, and return a list of each
  image's max softmax probability (its confidence in its own top guess).

  Works the same whether loader contains known-class images or OOD images. The model doesn't "know" which kind it's looking at, it just outputs
  probabilities either way.
  """
  confidences = []

  with torch.no_grad():
    for images, labels in loader:
      images = images.to(device)

      outputs = model(images)
      probs = torch.softmax(outputs, dim=1)
      max_probs, _ = probs.max(dim=1)

      confidences.extend(max_probs.tolist())

  return confidences

if __name__ == '__main__':
  ood_dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=ood_class_to_idx, transform=transform)
  ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=4)

  print(f"Total OOD samples: {len(ood_dataset)}")

  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  num_classes = 46
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, num_classes)

  state_dict = torch.load('models/resnet18_baseline.pt')
  model.load_state_dict(state_dict)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  model.eval()
  known_class_counts = get_class_counts(DATA_DIR)
  known_classes = filter_dataset(known_class_counts)
  known_class_to_idx = get_class_to_idx(known_classes)

  known_dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=known_class_to_idx, transform=transform)
  _, known_test_indices = split_dataset(known_dataset)
  known_test_dataset = Subset(known_dataset, known_test_indices)
  known_test_loader = DataLoader(known_test_dataset, batch_size=32, shuffle=False, num_workers=4)

  ood_confidences = get_msp_confidences(model, ood_loader, device)

  print(f"Number of OOD confidence values collected: {len(ood_confidences)}")
  print(f"Average OOD confidence: {sum(ood_confidences) / len(ood_confidences):.4f}")

  known_confidences = get_msp_confidences(model, known_test_loader, device)
  print(f"Average known-class test confidence: {sum(known_confidences) / len(known_confidences):.4f}")

  threshold = np.percentile(known_confidences, 5)
  print(f"Threshold (5th percentile of known confidences): {threshold:.4f}")
  ood_flagged = sum(1 for c in ood_confidences if c < threshold) / len(ood_confidences)
  known_flagged = sum(1 for c in known_confidences if c < threshold) / len(known_confidences)

  print(f"OOD images correctly flagged as novel: {ood_flagged:.4f}")
  print(f"Known images wrongly flagged as novel (false positive rate): {known_flagged:.4f}")