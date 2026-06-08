# Critical Infrastructure and Post-Disaster Damage Classification from Satellite Imagery

**MVP Report**  
**Team Members:** Owen Loughery, Carson Nuebert  
**Course:** DATA 402

## Overview

Our project trains machine learning models to classify satellite imagery into eight categories relevant for disaster response. The categories cover visible disaster damage (flooded roads, debris), critical infrastructure (hospitals, fire stations), and residential zones (single-unit, multi-unit, impoverished settlements), plus construction sites as a confounder class. The goal is to evaluate how much performance you can get out of progressively more capable model architectures on this satellite imagery, going from a linear baseline up to a fine-tuned pretrained CNN.

For the MVP we have all three planned models trained and evaluated on the test set, with the full pipeline running locally on a Windows laptop using a custom data download script and a Jupyter notebook for modeling.

## 1. Dataset and Preprocessing

### Source

We use the Functional Map of the World (fMoW) dataset, released by IARPA in 2018. The full dataset has over one million high-resolution satellite images across 63 categories from over 200 countries, hosted publicly on AWS S3. We are using the RGB version of the imagery.

### Sampling strategy

We selected 8 categories from the original 63 based on relevance to disaster response. We did not use all available data because the full dataset is over 200 GB, way more than what we needed for this project. Instead we wrote a custom Python pipeline (`data_loader.py`) that downloads and preprocesses a focused subset into an image folder collection:

For training, we sampled up to 1500 locations per class from fMoW's train split. Because the rare disaster classes (flooded_road, debris_or_rubble, impoverished_settlement, construction_site) have fewer than 1500 locations available, we used a hybrid sampling strategy: keep all available temporal views for the small classes (giving them more total images) and one view per location for the big classes (avoiding inflating their counts). This brought the imbalance ratio from a worst case of about 12:1 down to roughly 3.3:1.

For validation and test, we sampled 400 locations per class from fMoW's val split, took one image per location to avoid having near-duplicate views inflate eval metrics, and randomly split 50/50 into validation and test sets. Train, val, and test come from completely disjoint sets of geographic locations, so there is no data leakage.

### Final dataset sizes

| Split | Total Images | Smallest Class | Largest Class |
|-------|--------------|----------------|---------------|
| Train | 14,103 | impoverished_settlement (1,366) | construction_site (3,326) |
| Validation | 580 | impoverished_settlement (16) | single-unit_residential (200) |
| Test | 585 | impoverished_settlement (17) | single-unit_residential (200) |

### Preprocessing

Each raw image comes with a paired JSON metadata file that includes a bounding box around the labeled object plus information like ground sample distance, sun angle, and cloud cover. Our preprocessing pipeline:

1. Reads the bounding box from the JSON and crops the raw image to that region with 20% padding on each side
2. Resizes the crop to 224x224 pixels (standard input size for most CNN architectures)
3. Saves the result as a JPEG in a folder structure organized by class

For the logistic regression baseline we additionally downsample each 224x224 image to 64x64 to make the feature matrix manageable in memory (64x64x3 = 12,288 features per image vs 150,528 at full resolution).

For the neural network models, training images go through data augmentation at load time: random horizontal flips, random vertical flips, and random rotations up to 180 degrees. We can use aggressive augmentation here because satellite imagery has no canonical orientation, so a rotated or flipped image is still a valid example of the same class. Validation and test images do not get augmented. All images get normalized using ImageNet mean and standard deviation values to match what the pretrained ResNet expects.

### Class imbalance handling

We compute class weights inversely proportional to class frequency using the standard formula `n_samples / (n_classes * class_count)`. These weights get passed to sklearn's logistic regression as `class_weight='balanced'` and to PyTorch's `CrossEntropyLoss` via the `weight` parameter. This stops the models from learning to just predict the majority class.

## 2. Baseline Models

We implemented three models that span a clean progression from a linear baseline to a fine-tuned pretrained CNN.

### Model 1: Logistic Regression (sklearn)

Multinomial logistic regression on flattened RGB pixel values at 64x64 resolution (12,288 features per image). Features are z-score normalized using `StandardScaler` fit on training data. We swept regularization strength C across [0.001, 0.01, 0.1, 1.0, 10.0] and selected the best value (C=0.001) based on validation macro-F1. The strong regularization makes sense given the model has 12,288 features and only 14,103 training samples.

This model sets the floor for what a linear, spatially-naive approach can achieve.

### Model 2: From-scratch CNN (PyTorch)

A custom convolutional neural network with 4 conv-pool blocks. Each block has Conv2d (with 3x3 kernels and padding=1), BatchNorm2d, ReLU, then MaxPool2d. The channel progression is 3 to 32 to 64 to 128 to 256, with spatial dimensions shrinking from 224x224 down to 14x14 through the pooling layers. After the conv stack, AdaptiveAvgPool2d collapses the spatial dimensions to 1x1, then Dropout(0.5) and a final Linear(256, 8) produces the class scores.

Trained with Adam (lr=1e-3) and weighted cross-entropy loss for 30 epochs at batch size 32. Best model selected by validation macro-F1.

This model adds spatial structure awareness compared to LR but starts from random initialization, so all visual features have to be learned from scratch using only our 14k training images.

### Model 3: Pretrained ResNet-18 (PyTorch, transfer learning)

ResNet-18 with weights pretrained on ImageNet, with the final fully-connected layer replaced with a fresh Linear(512, 8) head for our 8 classes. Fine-tuned end-to-end with Adam at lr=1e-4 (smaller learning rate because the pretrained weights are already in a good area). Same loss function and best-model logic as the from-scratch CNN.

This model adds prior knowledge from a much larger image dataset on top of the spatial structure awareness, so it does not have to learn what edges and textures look like from scratch.

## 3. Evaluation

We use macro-averaged precision, recall, and F1 score as our primary metrics, plus per-class F1 scores and confusion matrices. Macro-averaging matters here because of class imbalance: it treats every class as equally important regardless of how many examples it has in the test set. Accuracy alone can be misleading when a model gets most of the majority class right but does poorly on minority classes.

We selected hyperparameters (C for LR, best epoch for CNN and ResNet) using only the validation set, then ran each finalized model on the test set exactly once to get the final reported numbers.

### Final test set results

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-------------|-----------------|--------------|
| Logistic Regression | 0.221 | 0.188 | 0.237 | 0.199 | 0.210 |
| From-scratch CNN | 0.552 | 0.505 | 0.543 | 0.527 | 0.556 |
| Pretrained ResNet-18 | **0.668** | **0.653** | **0.666** | **0.652** | **0.665** |

### Per-class F1 scores on test set

| Class | LR | CNN | ResNet |
|-------|-----|-----|--------|
| construction_site | 0.075 | 0.319 | **0.613** |
| debris_or_rubble | 0.055 | 0.319 | **0.435** |
| fire_station | 0.227 | 0.611 | **0.736** |
| flooded_road | 0.252 | 0.615 | **0.874** |
| hospital | 0.266 | 0.298 | **0.593** |
| impoverished_settlement | 0.100 | **0.824** | 0.750 |
| multi-unit_residential | 0.237 | 0.316 | **0.487** |
| single-unit_residential | 0.291 | **0.742** | 0.733 |

### Discussion

The three-tier story plays out exactly as predicted in the proposal. The from-scratch CNN improves macro-F1 by 2.7x over the LR baseline, showing the value of respecting spatial structure in image data. The pretrained ResNet adds another 29% improvement on top of that, showing the value of transfer learning when training data is limited (14k images is a fair amount but small compared to ImageNet's 1.2M).

The validation-to-test gap is small for all three models. Validation macro-F1 was 0.203 (LR), 0.536 (CNN), and 0.671 (ResNet), so test results are within 1 to 3 points of validation. This suggests we did not significantly overfit our hyperparameters to validation.

The flooded_road class is the highest-value win. ResNet hits F1=0.874 on it with recall=0.950, meaning it catches almost every flooded road in test. For a disaster-response use case this is the most important class to get right, since missing a flooded road is a more costly error than misclassifying a residential neighborhood.

## 4. Error Analysis

### Where models perform poorly

Hospital and multi-unit residential are the two classes that struggle across all three models. Hospital is particularly interesting because the ResNet has good precision (0.656) but lower recall (0.541), meaning when it predicts hospital it is usually right, but it misses about half of actual hospitals. From the confusion matrix, hospitals frequently get predicted as multi-unit residential or construction site, which makes sense visually because all three involve large multi-building footprints. This is genuine task difficulty, not a model flaw: separating a hospital from a generic large institutional building at 224x224 resolution is hard for any model that does not have access to context like signage or surrounding road networks.

Debris_or_rubble has the lowest absolute F1 across models, but this is partly a small-sample issue. The test set has only 19 debris examples, so one extra misclassification swings F1 by 5 to 6 percentage points.

### Overfitting evidence

The ResNet showed clear overfitting during training. By epoch 30, training accuracy hit 0.935 while validation accuracy stayed around 0.628, a 30 point gap. Validation loss climbed steadily after epoch 7 even as training loss kept falling. This is expected behavior when fine-tuning an 11 million parameter pretrained model on only 14k images: the model has enough capacity to memorize specific training examples faster than it can learn generalizable patterns.

Our save-best-model logic captured the optimal epoch (epoch 13) before overfitting got bad. In retrospect, we could have used early stopping to terminate training automatically once validation stopped improving, but the save-best approach gave us the same final result with less code.

The from-scratch CNN showed almost no overfitting. Train and validation accuracy tracked each other closely throughout the 30 epochs (0.554 vs 0.552 at the best epoch). This suggests the from-scratch model is closer to underfitting than overfitting: it has less capacity, so more training time or a larger architecture could potentially help.

The LR baseline also did not overfit, but for a different reason: the strong regularization (C=0.001) was specifically chosen to prevent the 12,288-feature model from overfitting its 14k samples.

### Limitations

The biggest limitation is dataset size. fMoW is a large dataset overall but the disaster categories specifically (flooded_road, debris_or_rubble, impoverished_settlement) have relatively few examples, and our test set has fewer than 20 examples each for the rarest classes. This makes per-class test metrics noisy.

A second limitation is that fMoW labels are at the location level, not the image level. Some flooded_road locations include images taken when the road was dry, which is genuine label noise for any single-image classifier. We accepted this for simplicity rather than building custom filtering logic to remove "dry" views of "flooded_road" locations.

A third limitation is that our preprocessing throws away rich metadata that comes with each image (sun angles, ground sample distance, country, timestamp). The original fMoW paper showed that combining image features with metadata improves accuracy. We did not pursue metadata fusion for this project because the comparison story across our three models is cleaner without it.

## 5. Progress Toward Final Project

### What currently works

We are essentially done with the modeling work. All three planned models are implemented, trained, evaluated on the validation set for hyperparameter selection, and evaluated on the held-out test set for final reporting. The full pipeline is reproducible: anyone can clone our GitHub repository, run `python data_loader.py` to download and preprocess the dataset to a local folder, then run the Jupyter notebook to retrain all three models and reproduce our test numbers.

### What we have planned for the final report

Polish on the existing analysis and presentation:

1. Cleaner visualizations for the final report (we have side-by-side confusion matrix heatmaps and grouped bar charts of per-class F1 already, but we want to refine them)
2. Deeper error analysis using the confusion matrices to call out specific class-pair confusions (hospital vs multi-unit residential, debris vs construction site) with example images
3. A short discussion of practical deployment considerations: which classes a real disaster-response system could rely on (flooded_road, single-unit_residential) and which would need more work (hospital, debris_or_rubble)
4. Showing what the features look like from the images

### Possible additional experiments if time permits

These are stretch goals, not requirements:

1. Adding learning rate scheduling (cosine annealing) to the from-scratch CNN to see if it pushes past 0.55 macro-F1
2. Trying ResNet-50 instead of ResNet-18 to see if the larger model helps, with the caveat that our 6 GB laptop GPU may be a memory bottleneck
3. Looking at whether ensemble predictions (averaging probabilities across the three models) improves over the best single model
4. Possibly doing a human test having us try and classify the test sets ourselves and seeing how accurate we can get it to compare the transfer learning model to our own prediction scores.

## Repository and Running Instructions

GitHub repository: https://github.com/OwenLoughery/FMOW-Disaster-Image-Classification

To run the project:

1. Clone the repository
2. Install dependencies: `pip install boto3 numpy pillow scikit-learn pandas matplotlib jupyter torch torchvision`
3. Run `python data_loader.py` from the project root to download and preprocess the dataset (takes 20 to 45 minutes one time). This pulls about 33,000 raw images from the public fMoW S3 bucket, crops and resizes them to 224x224, and saves them under `data/processed`, `data/processed_val`, and `data/processed_test`.
4. Open `fmow_modeling.ipynb` in Jupyter or PyCharm and run cells in order. The notebook handles all three models from loading data through final test evaluation.
