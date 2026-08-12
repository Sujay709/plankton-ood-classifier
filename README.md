# Plankton Classifier with Out-of-Distribution Detection

A convolutional neural network for classifying plankton imagery, extended with
an out-of-distribution (OOD) detection layer that flags organisms which don't
confidently match any known class, instead of forcing them into the nearest
known category.

## Motivation

Standard plankton classification models assume every image belongs to one of
a fixed set of trained classes. In practice, imaging datasets, especially ones
collected in situ, regularly contain rare, ambiguous, or previously undescribed
organisms. Forcing every image into the nearest known class risks silently
mislabeling these cases. This project treats "doesn't match a known class" as
a valid output in its own right.

## Planned Approach

- Baseline CNN (ResNet18 transfer learning) for multi-class plankton classification
- OOD scoring layer (max softmax probability baseline, possibly Mahalanobis distance)
  to identify likely novel or unknown organisms
- Evaluation on both held-out known classes and held-out "unknown" classes to
  assess detection performance

## Project Structure

```
data/         raw and processed plankton imagery
src/          dataset loading, model, training, OOD scoring, and evaluation
notebooks/    exploratory analysis
results/      model outputs and evaluation metrics
models/       saved model weights
```

## Data

This project uses annotated plankton imagery from the **WHOI-Plankton** dataset,
collected in situ via Imaging FlowCytobot (IFCB) at the Martha's Vineyard
Coastal Observatory (MVCO), Woods Hole Oceanographic Institution.

- 2013 subset: https://hdl.handle.net/1912/7349
- 2014 subset: https://hdl.handle.net/1912/7350

Initial exploration (`notebooks/01_eda.ipynb`) found 103 annotated classes with
substantial class imbalance and images that are grayscale, variably sized, and
low signal-to-noise. These findings shape the preprocessing and training design.

**Citation:**

> Sosik, H. M., Peacock, E. E., and Brownlee, E. F. (2015). Annotated Plankton
> Images - Data Set for Developing and Evaluating Classification Methods.
> https://doi.org/10.1575/1912/7341

```bibtex
@misc{whoiplankton,
  title={Annotated Plankton Images - Data Set for Developing and Evaluating Classification Methods},
  author={Sosik, Heidi M. and Peacock, Emily E. and Brownlee, Emily F.},
  year={2015},
  doi={10.1575/1912/7341},
  url={https://hdl.handle.net/1912/7341}
}
```

## Status

Early development. Data exploration is done. Classification model and OOD
detection aren't built yet.