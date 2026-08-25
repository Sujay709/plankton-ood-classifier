from pathlib import Path
import numpy as np
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import  Subset
from dataset import get_class_counts, filter_dataset, get_class_to_idx, split_dataset, PlanktonDataset, OOD_CLASSES
from model import build_model

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

def get_features(model, loader, device):
  """
  Run every image in loader through the model and grab the 512-dim feature
  vector each one produces right before the final classification layer.

  Uses a forward hook on avgpool to snatch those features mid-pass, since
  the model normally throws them away after turning them into class scores.

  Returns all the feature vectors stacked together, plus their true labels,
  so you can later compute things like class means and covariance.
  """
  features_storage = []
  def hook_fn(module, input, output):
    features_storage.clear()
    features_storage.append(output)

  handle = model.avgpool.register_forward_hook(hook_fn)
  model.eval()

  all_features = []
  all_labels = []
  with torch.no_grad():
    for images, labels in loader:
      images = images.to(device)
      model(images)
      batch_features = features_storage[0].flatten(1)
      batch_features = batch_features.cpu()
      labels = labels.cpu()

      all_features.append(batch_features)
      all_labels.append(labels)
  handle.remove()

  features = torch.cat(all_features, dim=0)
  labels = torch.cat(all_labels, dim=0)

  return features,labels


def get_class_means(features, labels, num_classes):
  """
  Work out the average feature vector for each class.

  For every class, grab all the feature vectors that belong to it and
  average them together. This gives you one 512-dim vector per class,
  basically a summary of what a typical image in that class looks like
  in feature space.
  """
  class_means = []
  for i in range(num_classes):
    mask = labels == i
    class_mean = features[mask].mean(dim=0)
    class_means.append(class_mean)
  return torch.stack(class_means)

def get_shared_covariance(features, labels, class_means, num_classes):
  """
  Figure out how features naturally vary and correlate with each other,
  pooling that info across all classes into one shared covariance matrix.

  For each class, subtract that class's mean from its own feature vectors
  to get pure deviations (how far each image strays from its own class's
  average). Stack all classes' deviations together, then compute the
  covariance across the whole pooled set.

  This shared covariance is what Mahalanobis distance uses later to figure
  out which directions in feature space are normal wobble vs which
  ones are actually suspicious.
  """
  all_deviations = []
  for i in range(num_classes):
    mask = labels == i
    deviations = features[mask] - class_means[i]
    all_deviations.append(deviations)
  combined_total_deviations = torch.cat(all_deviations, dim=0)
  swapped_dim_deviations = combined_total_deviations.T
  covariance_matrix = torch.cov(swapped_dim_deviations)

  return covariance_matrix

def mahalanobis_distance(features, class_means, covariance):
  """
  Work out how far each image sits from every class mean, using
  Mahalanobis distance instead of plain Euclidean distance.

  The covariance matrix gets a tiny value added to its diagonal first,
  just so it's guaranteed to be invertible. Its inverse is what lets us
  properly reweight each feature dimension: directions that naturally
  wobble a lot count for less, and directions that are normally stable
  count for more, so a deviation only looks "suspicious" if it's
  unusual relative to how that feature normally behaves.

  Returns a distance for every image against every class, so for N
  images and 46 classes you get back an (N, 46) grid of distances.
  """
  covariance_regularized = covariance + (1e-6 * torch.eye(512))
  cov_inv = torch.linalg.inv(covariance_regularized)
  features_unsq = features.unsqueeze(1)
  class_means_unsq = class_means.unsqueeze(0)
  deviations = features_unsq-class_means_unsq
  
  transformed_deviations = torch.einsum('ik,nck->nci', cov_inv, deviations)
  distance = torch.sqrt(torch.einsum('nci, nci->nc', transformed_deviations, deviations))

  return distance

if __name__ == '__main__':
  known_class_counts = get_class_counts(DATA_DIR)
  known_classes = filter_dataset(known_class_counts)
  known_class_to_idx = get_class_to_idx(known_classes)
  ood_dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=ood_class_to_idx, transform=transform)
  ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=4)

  print(f"Total OOD samples: {len(ood_dataset)}")

  known_dataset = PlanktonDataset(root_dir=DATA_DIR, class_to_idx=known_class_to_idx, transform=transform)
  known_train_indices, known_test_indices = split_dataset(known_dataset)
  known_train_dataset = Subset(known_dataset, known_train_indices)
  known_train_loader = DataLoader(known_train_dataset, batch_size=32, shuffle=False, num_workers=4)
  known_test_dataset = Subset(known_dataset, known_test_indices)
  known_test_loader = DataLoader(known_test_dataset, batch_size=32, shuffle=False, num_workers=4)
  
  num_classes = len(known_class_to_idx)
  model = build_model(num_classes)

  state_dict = torch.load('models/resnet18_baseline.pt')
  model.load_state_dict(state_dict)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  model.eval()


  train_features, train_labels = get_features(model, known_train_loader, device)

  test_features, test_labels = get_features(model, known_test_loader, device)
  ood_features, ood_labels = get_features(model, ood_loader, device)

  class_means = get_class_means(train_features, train_labels, num_classes)
  covariance = get_shared_covariance(train_features, train_labels, class_means, num_classes)
  test_distances = mahalanobis_distance(test_features, class_means, covariance)
  ood_distances = mahalanobis_distance(ood_features, class_means, covariance)

  test_min_distances, _ = test_distances.min(dim=1)
  ood_min_distances, _ = ood_distances.min(dim=1)

  print(f"Average known-class test distance: {test_min_distances.mean().item():.4f}")
  print(f"Average OOD distance: {ood_min_distances.mean().item():.4f}")

  mahalanobis_threshold = np.percentile(test_min_distances.tolist(), 95)
  
  mahalanobis_ood_flagged = sum(1 for c in ood_min_distances if c > mahalanobis_threshold) / len(ood_min_distances)
  mahalanobis_known_flagged = sum(1 for c in test_min_distances if c > mahalanobis_threshold) / len(test_min_distances)

  print(f"Threshold (95th percentile of known distances): {mahalanobis_threshold:.4f}")
  print(f"OOD images correctly flagged as novel: {mahalanobis_ood_flagged:.4f}")
  print(f"Known images wrongly flagged as novel (false positive rate): {mahalanobis_known_flagged:.4f}")
  
  ood_confidences = get_msp_confidences(model, ood_loader, device)

  print(f"Number of OOD confidence values collected: {len(ood_confidences)}")
  print(f"Average OOD confidence: {sum(ood_confidences) / len(ood_confidences):.4f}")

  known_confidences = get_msp_confidences(model, known_test_loader, device)
  print(f"Average known-class test confidence: {sum(known_confidences) / len(known_confidences):.4f}")

  msp_threshold = np.percentile(known_confidences, 5)
  print(f"Threshold (5th percentile of known confidences): {msp_threshold:.4f}")
  ood_flagged = sum(1 for c in ood_confidences if c < msp_threshold) / len(ood_confidences)
  known_flagged = sum(1 for c in known_confidences if c < msp_threshold) / len(known_confidences)

  print(f"OOD images correctly flagged as novel: {ood_flagged:.4f}")
  print(f"Known images wrongly flagged as novel (false positive rate): {known_flagged:.4f}")

  results = {
    "mahalanobis": {
      "avg_known_distance": test_min_distances.mean().item(),
      "avg_ood_distance": ood_min_distances.mean().item(),
      "threshold": float(mahalanobis_threshold),
      "ood_detection_rate": mahalanobis_ood_flagged,
      "false_positive_rate": mahalanobis_known_flagged,
    },
    "msp": {
      "avg_known_confidence": sum(known_confidences) / len(known_confidences),
      "avg_ood_confidence": sum(ood_confidences) / len(ood_confidences),
      "threshold": float(msp_threshold),
      "ood_detection_rate": ood_flagged,
      "false_positive_rate": known_flagged,
    },
  }
  Path('results').mkdir(exist_ok=True)
  with open('results/ood_results.json', 'w') as f:
    json.dump(results, f, indent=2)
  


