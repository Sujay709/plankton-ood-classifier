from pathlib import Path

DATA_DIR = Path('data/raw/2013')

def get_class_counts(data_dir):
  """Count images per class folder under data_dir."""
  class_counts = {}
  for class_folder in data_dir.iterdir():
    if class_folder.is_dir():
      count = len(list(class_folder.glob('*.png')))
      class_counts[class_folder.name] = count
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

if __name__ == '__main__':
  class_counts = get_class_counts(DATA_DIR)
  known_classes = filter_dataset(class_counts)
  class_to_idx = get_class_to_idx(known_classes)
  print(f"classes length: {len(known_classes)}", end="\n\n")
  print(f"known classes: {known_classes}", end="\n\n")
  print(f"class to idx: {class_to_idx}", end="\n\n")