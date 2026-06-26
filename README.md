# Post-Disaster Damage and Infrastructure Classification from Satellite Imagery

A multi-class image classification project that automatically sorts satellite image tiles into 8 categories relevant to disaster response. Built as class project at Cal Poly SLO.

When a disaster hits, responders need to know where the damage is and where critical resources are located. That work is mostly done by hand today, labeling satellite tiles one at a time, which does not scale when an event affects a whole region overnight. This project trains models to do that classification automatically.

The core idea is a clean three-tier comparison where each model adds exactly one capability over the previous one, so we can measure what each capability is actually worth:

| Model | Adds | Test Macro-F1 |
|-------|------|---------------|
| Logistic Regression (flattened pixels) | baseline | 0.188 |
| From-scratch CNN | spatial structure awareness | 0.505 |
| Pretrained ResNet-18 (fine-tuned) | ImageNet prior knowledge | **0.653** |

The CNN improves macro-F1 by 2.7x over the linear baseline, and pretraining adds another ~29% on top of that.

## Classes

8 categories selected from the 63 in fMoW, grouped by function:

- **Visible damage:** `flooded_road`, `debris_or_rubble`
- **Critical infrastructure:** `hospital`, `fire_station`
- **Population zones:** `single-unit_residential`, `multi-unit_residential`, `impoverished_settlement`
- **Confounder:** `construction_site` (visually resembles debris, forces the model to learn real discriminative features)

## Dataset

[Functional Map of the World (fMoW)](https://github.com/fMoW/dataset), RGB version, released by IARPA in 2018. The full dataset is over 1 million images and 200GB+, hosted publicly on AWS S3 with anonymous access.

Rather than download all of it, `data_loader.py` pulls only the images needed for the 8 chosen classes. Key pipeline decisions:

- **Sampling:** hybrid strategy. Rare disaster classes keep all temporal views per location; common classes keep one view per location. This brings the training imbalance from ~12:1 down to ~3.3:1.
- **Leakage prevention:** train comes from fMoW's train split, val/test from its val split, so the two share no geographic locations.
- **Preprocessing:** each image is cropped to its labeled bounding box (with 20% padding) using the paired JSON metadata, then resized to 224x224. Logistic regression additionally downsamples to 64x64 to keep the feature matrix manageable.
- **Augmentation:** random flips and rotations at load time (satellite imagery has no canonical orientation).
- **Normalization:** ImageNet mean/std, since the final model uses ImageNet pretrained weights.

Final sizes: ~14,100 train / ~580 val / ~585 test images.

## Repository structure

| File | Description |
|------|-------------|
| `data_loader.py` | End-to-end pipeline: downloads from S3, crops/resizes, saves to a local `data/` folder, and exposes load functions the notebook imports. |
| `DisasterClassifier.ipynb` | Main notebook. Trains and evaluates all three models with a shared train/val/test discipline. |
| `Final_Report.md` | Full writeup: methodology, model architectures, per-class results, and analysis. |
| `MVP.md` | Earlier minimum-viable-product milestone. |
| `best_cnn.pth` / `best_resnet.pth` | Saved weights from the best validation epoch for each model. |
| `comparison_*.png` | Result figures (macro-F1 by model, per-class F1, confusion matrices). |

## Usage

First time (downloads and preprocesses everything, takes ~20–45 minutes):

```bash
python data_loader.py
```

Then open `DisasterClassifier.ipynb` and run the cells to train and evaluate. The notebook imports the load functions from `data_loader.py` directly, so subsequent runs reuse the local `data/` folder without re-downloading.

**Requirements:** Python with PyTorch, torchvision, scikit-learn, boto3, numpy, and matplotlib. A CUDA GPU is recommended for the CNN and ResNet.

## Methodology notes

All three models follow the same discipline: hyperparameters tuned and checkpoints selected on the validation set only, then evaluated exactly once on the held-out test set. Macro-averaged F1 is the primary metric since it weights every class equally regardless of size, which matters given the class imbalance. Class imbalance is handled with weights inversely proportional to class frequency, passed to sklearn's `class_weight='balanced'` and PyTorch's `CrossEntropyLoss`.

## Authors

Owen Loughery and Carson Nuebert
