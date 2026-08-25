from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import json

import torchvision.transforms as transforms

from dataset import get_class_counts, filter_dataset, get_class_to_idx, split_dataset, PlanktonDataset
from model import build_model

DATA_DIR = Path("data/raw")

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_counts = get_class_counts(DATA_DIR)
known_classes = filter_dataset(class_counts)
class_to_idx = get_class_to_idx(known_classes)
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

if __name__ == '__main__':
  dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=class_to_idx, transform=transform)
  train_indices, test_indices = split_dataset(dataset)
  
  test_dataset = Subset(dataset, test_indices)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

  num_classes = len(class_to_idx)
  model = build_model(num_classes)

  state_dict = torch.load('models/resnet18_baseline.pt')
  model.load_state_dict(state_dict)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  model.eval()

  correct = 0
  total = 0
  class_correct = {}
  class_total = {}
  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)

      preds = torch.argmax(outputs, dim=1)
      correct += (preds == labels).sum().item()
      for pred, true_label in zip(preds.tolist(), labels.tolist()):
        class_total[true_label] = class_total.get(true_label, 0) + 1
        if pred == true_label:
          class_correct[true_label] = class_correct.get(true_label, 0) + 1

      total += labels.size(0)
      accuracy = correct / total
  
  print(f"Test accuracy: {accuracy:.4f}")
  per_class_accuracy = {}
  for class_name, idx in class_to_idx.items():
    per_class_accuracy[class_name] = class_correct.get(idx, 0) / class_total.get(idx, 1)
  results = {"overall_accuracy": accuracy, "per_class_accuracy": per_class_accuracy}
  Path('results').mkdir(exist_ok=True)
  with open('results/evaluate_results.json', 'w') as f:
    json.dump(results, f, indent=2)