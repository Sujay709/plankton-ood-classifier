# Plankton Classifier with Novel-Species Detection

A ResNet18-based image classifier for plankton species, extended with a novel-species detection layer that flags organisms outside the known training classes instead of forcing them into the wrong bucket.

## Motivation

Plankton imaging systems like IFCB collect far more organisms than any fixed label set can cover. A classifier that misclassifies a new or rare species as one of its known classes is useless for downstream ecological analysis. This project treats novel-species detection as a first-class part of the pipeline, not an afterthought.

## Dataset

WHOI-Plankton dataset (Sosik, Peacock, Brownlee 2015), IFCB imagery from the Martha's Vineyard Coastal Observatory.

- 2013 subset: hdl.handle.net/1912/7349
- 2014 subset: hdl.handle.net/1912/7350

Images are grayscale, variable size and aspect ratio, converted to RGB via channel replication for compatibility with ImageNet-pretrained weights.

## Pipeline

1. **Filtering**: classes with fewer than 50 images are dropped, along with junk/catchall labels (`mix`, `detritus`, `bad`, `other_interaction`). 46 classes remain for training.
2. **Held-out OOD classes**: 9 species (Euglena, Dictyocha, Thalassionema, Ditylum, Pyramimonas_longicauda, Proterythropsis_sp, Chaetoceros_pennate, Tontonia_gracillima, Guinardia_striata) are excluded entirely from training and used only to test novel-species detection.
3. **Baseline classifier**: ResNet18 pretrained on ImageNet, backbone frozen, final layer retrained on the 46 known classes.
4. **Novel-species detection**: two methods, compared head to head.
   - **MSP (Maximum Softmax Probability)**: flags an image as novel if the model's top confidence falls below a threshold set from the known-class validation distribution.
   - **Mahalanobis distance**: fits a class-conditional Gaussian in the model's penultimate feature space (per-class means, shared covariance) and flags an image as novel if it's too far from every class centroid.

## Results

Baseline classification accuracy on held-out known-class test data: **80.18%**

Novel-species detection, both methods calibrated to a 5% false positive rate on known classes:

| Method | OOD detection rate |
|---|---|
| MSP | 22.35% |
| Mahalanobis | 35.70% |

Mahalanobis distance catches over 1.5x as many novel species as MSP at the same false alarm rate, consistent with MSP being a known weak baseline in the OOD detection literature.

## Repo structure

```
data/raw/          # 2013 and 2014 image subfolders (not tracked)
data/processed/    # (not tracked)
models/            # saved model weights (not tracked)
notebooks/         # EDA
results/           # JSON output from evaluate.py and ood.py
src/
  dataset.py       # data loading, filtering, class mapping, train/test split
  model.py         # ResNet18 construction
  train.py         # baseline training loop
  evaluate.py      # test accuracy, per-class accuracy
  ood.py           # MSP and Mahalanobis novel-species detection
```

## Running it

```
python src/train.py      # trains baseline, saves models/resnet18_baseline.pt
python src/evaluate.py   # test accuracy + per-class breakdown -> results/evaluate_results.json
python src/ood.py        # MSP + Mahalanobis novel-species detection -> results/ood_results.json
```

## Notes

This started as a way to build hands-on experience with a lab-relevant pipeline (image classification on underwater/microscopy imagery) rather than a direct reproduction of any specific published work. The novel-species detection layer is the original contribution on top of a standard transfer-learning baseline.