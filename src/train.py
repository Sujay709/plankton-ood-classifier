import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from dataset import get_class_counts, filter_dataset, get_class_to_idx, PlanktonDataset
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

if __name__ == '__main__':
  train_dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=class_to_idx, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

  num_classes = len(class_to_idx)
  model = build_model(num_classes)

  for name, param in model.named_parameters():
    if not name.startswith('fc'):
      param.requires_grad = False

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  print(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

  num_epochs = 5

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if i % 50 == 0:
        print(f"  batch {i}/{len(train_loader)}, loss so far: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

  torch.save(model.state_dict(), f='models/resnet18_baseline.pt')