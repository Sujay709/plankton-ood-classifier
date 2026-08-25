import torch.nn as nn
import torchvision.models as models

def build_model(num_classes):
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, num_classes)

  return model