from pathlib import Path
from PIL import Image


DATA_DIR = Path('data/raw')

def get_class_counts(data_dir):
  """Count images per class folder under data_dir."""
  class_counts = {}
  for year in ["2013", "2014"]:
    year_path = data_dir / year
    for class_folder in year_path.iterdir():
      if class_folder.is_dir():
        count = len(list(class_folder.glob('*.png')))
        class_counts[class_folder.name] = class_counts.get(class_folder.name, 0) + count
  return class_counts

def filter_dataset(class_counts):
  """
  Return the sorted list of class names to use for training.

  Excludes:
  - classes with fewer than MIN_CLASS_COUNT images (too little data to learn from)
  - junk/catchall classes (not real taxonomic categories)
  - held-out classes reserved for novel-species OOD evaluation
  """

  MIN_CLASS_COUNT = 50
  JUNK_CLASSES =  {'mix', 'detritus', 'bad', 'other_interaction'}
  OOD_CLASSES = {'Euglena', 'Dictyocha', 'Thalassionema', 'Ditylum', 'Pyramimonas_longicauda',
  'Proterythropsis_sp', 'Chaetoceros_pennate', 'Tontonia_gracillima', 'Guinardia_striata'}

  known_classes = [class_name for class_name, count in  class_counts.items() if count >= MIN_CLASS_COUNT and class_name not in JUNK_CLASSES and class_name not in OOD_CLASSES]

  return sorted(known_classes)

def get_class_to_idx(known_classes):
  """Map each class name to a stable integer index, based on sorted order."""
  class_to_idx = {}
  for idx, class_name in enumerate(known_classes):
    class_to_idx[class_name] = idx
  
  return class_to_idx

class PlanktonDataset:
  def __init__(self, root_dir, class_to_idx, transform=None):
    """
    Build the list of (image_path, label_idx) samples for all known classes
    across both years of data.

    Args:
      root_dir: Path to the root data directory containing year subfolders
      (e.g. 'data/raw'), each containing class-name subfolders of images.
      class_to_idx: Mapping from class name to integer label index.
      transform: Optional torchvision transform to apply to each image
      when it's loaded in __getitem__.
    """
    self.class_to_idx = class_to_idx 
    self.transform = transform 
    self.root_dir = Path(root_dir)

    self.samples = []
    for year in (["2013", "2014"]):
      for class_name, label_idx in class_to_idx.items():
        folder_path = self.root_dir / year / class_name
        if folder_path.exists():
          image_files = folder_path.glob("*.png")
          for image_path in image_files:
            self.samples.append((image_path, label_idx))

  def __len__(self):
    """Return the total number of samples in the dataset."""
    return len(self.samples)

  def __getitem__(self, idx):
    """
    Load and return a single (image, label) pair.

    Args:
        idx: Index into self.samples.

    Returns:
        A tuple of (image, label_idx), where image is a PIL Image converted
        to RGB (with any transform applied) and label_idx is an int.
    """
    image_path, label_idx = self.samples[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image, label_idx

if __name__ == '__main__':
  class_counts = get_class_counts(DATA_DIR)
  known_classes = filter_dataset(class_counts)
  class_to_idx = get_class_to_idx(known_classes)
  print(f"classes length: {len(known_classes)}", end="\n\n")
  print(f"known classes: {known_classes}", end="\n\n")
  print(f"class to idx: {class_to_idx}", end="\n\n")
  dataset = PlanktonDataset(root_dir='data/raw', class_to_idx=class_to_idx)
  print(f"total samples: {len(dataset.samples)}")
  print(f"first sample: {dataset.samples[0]}")
  print(f"len(dataset): {len(dataset)}")
  for i in [0, 44000, len(dataset) - 1]:
    image, label = dataset[i]
    path, stored_label = dataset.samples[i]
    print(f"idx={i} | mode={image.mode} | size={image.size} | label={label} | path={path}")