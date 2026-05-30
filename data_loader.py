"""
Data loader for the fMoW disaster classification project.

Reads preprocessed 224x224 satellite imagery from local disk into numpy
arrays suitable for sklearn baselines. The same data can be reused for
PyTorch models later via torchvision.datasets.ImageFolder pointed at the
same folders.

Expected folder layout (relative to project root):

    data/
        processed/              <- training set
            flooded_road/*.jpg
            debris_or_rubble/*.jpg
            ...
        processed_val/          <- validation set
            ...
        processed_test/         <- test set
            ...

Usage from a notebook:

    from data_loader import (
        load_all_splits, verify_data,
        CLASSES, SORTED_CLASSES, DATA_DIR,
    )
    verify_data()                                  # sanity check
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits(size=64)
"""

import os
import glob
import time
import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# The 8 disaster-response classes.
CLASSES = [
    'flooded_road',
    'debris_or_rubble',
    'hospital',
    'fire_station',
    'single-unit_residential',
    'multi-unit_residential',
    'impoverished_settlement',
    'construction_site',
]

# Alphabetical ordering matches torchvision.datasets.ImageFolder convention.
# Index in this list = numeric label used in y arrays. Always use
# SORTED_CLASSES[label] to convert back to a human-readable name.
SORTED_CLASSES = sorted(CLASSES)

# Default data location, relative to wherever the notebook is run from.
# Override by passing a different data_dir to load_all_splits().
DATA_DIR = 'data'


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_dataset(folder, classes=None, size=64, retries=3, verbose=True):
    """Load images from a class-structured folder into a flat numpy matrix.

    Each class lives in its own subfolder. The folder name is the class
    label; the numeric label is the position of that folder in `classes`.

    Args:
        folder:  path containing one subfolder per class
        classes: list of class names; defaults to SORTED_CLASSES
        size:    target square size in pixels (image is resized to size x size)
        retries: number of times to retry a failed file read
        verbose: print per-class load counts as it goes

    Returns:
        X: uint8 array of shape (n, size * size * 3), one row per image
        y: int array of shape (n,), integer class labels
    """
    if classes is None:
        classes = SORTED_CLASSES

    X, y = [], []
    skipped = 0

    for label_idx, cls in enumerate(classes):
        files = sorted(glob.glob(os.path.join(folder, cls, '*.jpg')))
        loaded = 0
        for f in files:
            for attempt in range(retries):
                try:
                    img = Image.open(f).convert('RGB').resize((size, size))
                    X.append(np.asarray(img, dtype=np.uint8).flatten())
                    y.append(label_idx)
                    loaded += 1
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(0.2)
                    else:
                        skipped += 1
        if verbose:
            print(f'  {cls:<26} {loaded:>6}')

    if skipped:
        print(f'  (skipped {skipped} unreadable files)')

    return np.array(X, dtype=np.uint8), np.array(y)


def load_all_splits(data_dir=None, size=64):
    """Convenience: load train, val, and test splits in one call.

    Args:
        data_dir: root data directory; defaults to DATA_DIR ('data')
        size:     target image size for each split

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    print('Training set:')
    X_train, y_train = load_dataset(os.path.join(data_dir, 'processed'), size=size)

    print('Validation set:')
    X_val, y_val = load_dataset(os.path.join(data_dir, 'processed_val'), size=size)

    print('Test set:')
    X_test, y_test = load_dataset(os.path.join(data_dir, 'processed_test'), size=size)

    print(f'\nShapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}')
    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------------------------------------------------------
# Sanity checking
# -----------------------------------------------------------------------------

def verify_data(data_dir=None):
    """Print per-class image counts for each split.

    Run this after downloading and extracting the data, before doing any
    modeling, to confirm everything is where it should be.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    splits = ['processed', 'processed_val', 'processed_test']
    print(f'Checking data in: {os.path.abspath(data_dir)}\n')

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            print(f'{split}: MISSING ({split_path} does not exist)')
            continue

        print(f'{split}:')
        total = 0
        for cls in SORTED_CLASSES:
            cls_path = os.path.join(split_path, cls)
            n = len(glob.glob(os.path.join(cls_path, '*.jpg'))) if os.path.isdir(cls_path) else 0
            total += n
            print(f'  {cls:<26} {n:>6}')
        print(f'  {"total":<26} {total:>6}\n')


if __name__ == '__main__':
    # Running this file directly performs the data check.
    verify_data()