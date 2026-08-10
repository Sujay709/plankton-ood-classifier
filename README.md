# Plankton Classifier with Out-of-Distribution Detection

A convolutional neural network for classifying plankton imagery, extended with
an out-of-distribution (OOD) detection layer that flags organisms which do not
confidently match any known class, rather than forcing them into the nearest
known category.

## Motivation

Standard plankton classification models assume every image belongs to one of
a fixed set of trained classes. In practice, imaging datasets — particularly
those collected in situ — regularly contain rare, ambiguous, or previously
undescribed organisms. Treating every image as forced-choice classification
risks silently mislabeling these cases. This project instead treats "does not
match a known class" as a valid and useful output, with the goal of supporting
applications such as biodiversity monitoring where flagging anomalies is often
more scientifically valuable than assigning an incorrect label.

## Approach

- Baseline CNN trained for multi-class plankton classification
- OOD scoring layer to identify likely novel or unknown organisms
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

In development. Data pipeline is in place; classification and OOD detection
components are in progress.