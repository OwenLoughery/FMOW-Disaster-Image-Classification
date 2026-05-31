"""
data_loader.py
==============

End-to-end data pipeline for the fMoW disaster classification project.

This single file handles:

1. Downloading raw satellite imagery from the public fMoW S3 bucket
   (no AWS credentials required — anonymous access).
2. Preprocessing each image (bounding-box crop, resize to 224x224).
3. Saving the processed images into a local `data/` folder structured
   for sklearn / torchvision use.
4. Providing load functions the modeling notebook imports.

Sampling strategy mirrors the Colab pipeline:
- Training: hybrid — small classes (rare disaster categories) keep all
  temporal views per location, big classes keep one view per location.
  Yields ~14,100 images at roughly 3.3:1 imbalance.
- Validation/Test: one view per location from fMoW's official val split,
  shuffled and split 50/50. Yields ~580 / ~585 images.

Usage
-----
First time (download + preprocess everything; takes ~20–45 minutes):

    python data_loader.py

Subsequent uses (in your notebook):

    from data_loader import load_all_splits, verify_data
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits(size=64)

Requirements: numpy, pillow, boto3, scikit-learn (for modeling, not here).

    pip install boto3 pillow numpy
"""

import os
import glob
import json
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

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
SORTED_CLASSES = sorted(CLASSES)

# Small (rare) classes — keep all temporal views per location.
# Big classes — one view per location.
SMALL = {'flooded_road', 'debris_or_rubble', 'impoverished_settlement', 'construction_site'}

# Folder layout (relative to project root):
DATA_DIR     = 'data'
RAW_DIR      = os.path.join(DATA_DIR, 'raw')             # temp, can delete after preprocess
OUT_TRAIN    = os.path.join(DATA_DIR, 'processed')
OUT_VAL      = os.path.join(DATA_DIR, 'processed_val')
OUT_TEST     = os.path.join(DATA_DIR, 'processed_test')

# Image preprocessing.
SIZE = 224          # output image side length in pixels
PAD  = 0.2          # bounding-box padding (fraction of box size)

# Sampling.
N_PER_CLASS_TRAIN = 1500     # locations per class for training
N_PER_CLASS_VAL   = 400      # locations per class drawn from fMoW val (split 50/50)

# S3 bucket configuration (public, anonymous).
S3_BUCKET = 'spacenet-dataset'
S3_BASE   = 'Hosted-Datasets/fmow/fmow-rgb'


# =============================================================================
# S3 client (lazy import so notebooks that only call load_* don't need boto3)
# =============================================================================

def _get_s3_client():
    """Return an anonymous boto3 client for fMoW's public bucket."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as e:
        raise ImportError(
            "boto3 is required for downloading the dataset. Install it with:\n"
            "    pip install boto3"
        ) from e
    return boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED, max_pool_connections=32),
    )


def _list_class_objects(s3, split, cls):
    """Paginated S3 listing for one (split, class). Returns {instance_folder: [keys]}."""
    prefix = f'{S3_BASE}/{split}/{cls}/'
    paginator = s3.get_paginator('list_objects_v2')
    by_inst = {}
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            k = obj['Key']
            if k.endswith('_rgb.jpg') or k.endswith('_rgb.json'):
                inst = k[len(prefix):].split('/')[0]
                by_inst.setdefault(inst, []).append(k)
    return by_inst


def _download_one(s3, key):
    """Download a single S3 key into RAW_DIR, mirroring its prefix structure.
    Skips files already present (so downloads are resumable)."""
    local = os.path.join(RAW_DIR, key[len(S3_BASE) + 1:])
    if os.path.exists(local):
        return
    os.makedirs(os.path.dirname(local), exist_ok=True)
    s3.download_file(S3_BUCKET, key, local)


def _download_keys(s3, keys, label, workers=32):
    """Threaded download of a list of S3 keys, with progress prints."""
    print(f'Downloading {len(keys)} files for {label}...')
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_download_one, s3, k) for k in keys]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                failed += 1
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{len(keys)}')
    print(f'  finished {label}: {done - failed} ok, {failed} failed\n')


# =============================================================================
# Image preprocessing
# =============================================================================

def _process_image(jpg_path, json_path, out_path):
    """Read raw fMoW jpg + json, crop to bounding box (with padding),
    resize to SIZE x SIZE, save."""
    with open(json_path) as f:
        meta = json.load(f)
    x, y, w, h = meta['bounding_boxes'][0]['box']
    img = Image.open(jpg_path).convert('RGB')
    px, py = int(w * PAD), int(h * PAD)
    crop = img.crop((
        max(0, x - px),
        max(0, y - py),
        min(img.width,  x + w + px),
        min(img.height, y + h + py),
    ))
    crop.resize((SIZE, SIZE)).save(out_path, quality=90)


# =============================================================================
# Pipeline: training set
# =============================================================================

def _download_train(s3):
    """Sample and download raw training images, hybrid by class."""
    print('=== TRAINING SET DOWNLOAD ===\n')
    random.seed(42)

    all_keys = []
    print(f"{'class':<26}{'locations':>10}{'files':>10}")
    for cls in CLASSES:
        by_inst = _list_class_objects(s3, 'train', cls)
        chosen = random.sample(list(by_inst), min(N_PER_CLASS_TRAIN, len(by_inst)))
        keys = [k for inst in chosen for k in by_inst[inst]]   # all temporal views
        all_keys += keys
        n_imgs = sum(1 for k in keys if k.endswith('.jpg'))
        print(f'{cls:<26}{len(chosen):>10}{n_imgs:>10}')

    print()
    _download_keys(s3, all_keys, 'training')


def _preprocess_train():
    """Crop + resize raw training images, save to OUT_TRAIN with hybrid view selection."""
    print('=== TRAINING SET PREPROCESSING ===\n')

    def select_jpgs(cls):
        all_jpgs = glob.glob(os.path.join(RAW_DIR, 'train', cls, '**', '*_rgb.jpg'), recursive=True)
        if cls in SMALL:
            return all_jpgs                              # all temporal views
        by_loc = defaultdict(list)                       # one view per location
        for j in all_jpgs:
            by_loc[os.path.dirname(j)].append(j)
        return [sorted(v)[0] for v in by_loc.values()]

    for cls in CLASSES:
        out_cls = os.path.join(OUT_TRAIN, cls)
        os.makedirs(out_cls, exist_ok=True)
        jpgs = select_jpgs(cls)
        n_done, n_skip = 0, 0
        for jpg in jpgs:
            out_path = os.path.join(out_cls, os.path.basename(jpg))
            if os.path.exists(out_path):
                n_done += 1
                continue
            try:
                _process_image(jpg, jpg.replace('.jpg', '.json'), out_path)
                n_done += 1
            except Exception:
                n_skip += 1
        skip_note = f' ({n_skip} skipped)' if n_skip else ''
        print(f'  {cls:<26} {n_done:>6}{skip_note}')
    print()


# =============================================================================
# Pipeline: val + test sets
# =============================================================================

def _download_val_test(s3):
    """Sample and download raw images from fMoW's val split (one view per location)."""
    print('=== VAL/TEST SET DOWNLOAD ===\n')
    random.seed(123)

    all_keys = []
    print(f"{'class':<26}{'locations':>10}{'images':>10}")
    for cls in CLASSES:
        by_inst = _list_class_objects(s3, 'val', cls)
        chosen = random.sample(list(by_inst), min(N_PER_CLASS_VAL, len(by_inst)))
        keys = []
        for inst in chosen:
            jpgs = sorted(k for k in by_inst[inst] if k.endswith('.jpg'))
            if not jpgs:
                continue
            first_jpg = jpgs[0]
            keys.append(first_jpg)
            keys.append(first_jpg.replace('.jpg', '.json'))
        all_keys += keys
        print(f'{cls:<26}{len(chosen):>10}{len(keys)//2:>10}')

    print()
    _download_keys(s3, all_keys, 'val/test')


def _preprocess_val_test():
    """Preprocess val raws, split 50/50 into OUT_VAL and OUT_TEST."""
    print('=== VAL/TEST SET PREPROCESSING (50/50 split) ===\n')
    random.seed(456)

    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_VAL,  cls), exist_ok=True)
        os.makedirs(os.path.join(OUT_TEST, cls), exist_ok=True)

        jpgs = sorted(glob.glob(
            os.path.join(RAW_DIR, 'val', cls, '**', '*_rgb.jpg'), recursive=True
        ))
        random.shuffle(jpgs)
        half = len(jpgs) // 2
        val_jpgs, test_jpgs = jpgs[:half], jpgs[half:]

        for jpg in val_jpgs:
            out_path = os.path.join(OUT_VAL, cls, os.path.basename(jpg))
            if os.path.exists(out_path):
                continue
            try:
                _process_image(jpg, jpg.replace('.jpg', '.json'), out_path)
            except Exception:
                pass

        for jpg in test_jpgs:
            out_path = os.path.join(OUT_TEST, cls, os.path.basename(jpg))
            if os.path.exists(out_path):
                continue
            try:
                _process_image(jpg, jpg.replace('.jpg', '.json'), out_path)
            except Exception:
                pass

        print(f'  {cls:<26}  val={len(val_jpgs):>4}  test={len(test_jpgs):>4}')
    print()


# =============================================================================
# Pipeline driver — checks what's done, fills in what's missing
# =============================================================================

def _split_is_complete(split_dir, min_per_class=50):
    """Heuristic: a processed split is 'done' if every class has at least
    min_per_class jpgs. Used to skip re-downloading if you've already run this."""
    if not os.path.isdir(split_dir):
        return False
    for cls in CLASSES:
        cls_path = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_path):
            return False
        if len(glob.glob(os.path.join(cls_path, '*.jpg'))) < min_per_class:
            return False
    return True


def setup_data(force=False):
    """Full pipeline: download from S3, preprocess, save to data/ folder.

    Resumable and idempotent — re-running it skips work that's already done.
    Pass force=True to redo preprocessing even if outputs exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR,  exist_ok=True)

    train_done = _split_is_complete(OUT_TRAIN) and not force
    valtest_done = (_split_is_complete(OUT_VAL) and _split_is_complete(OUT_TEST)) and not force

    if train_done and valtest_done:
        print('All splits already present. Nothing to do.\n')
        verify_data()
        print('\nIf you want to redo the pipeline anyway, call setup_data(force=True).')
        return

    s3 = _get_s3_client()
    start = time.time()

    if not train_done:
        _download_train(s3)
        _preprocess_train()
    else:
        print('Training set already processed — skipping.\n')

    if not valtest_done:
        _download_val_test(s3)
        _preprocess_val_test()
    else:
        print('Val/test sets already processed — skipping.\n')

    elapsed = (time.time() - start) / 60
    print(f'Pipeline complete in {elapsed:.1f} min.\n')
    print('You can safely delete the raw download folder to reclaim disk space:')
    print(f'    {os.path.abspath(RAW_DIR)}\n')
    verify_data()


# =============================================================================
# Load functions (used by the modeling notebook)
# =============================================================================

def load_dataset(folder, classes=None, size=64, retries=3, verbose=True):
    """Load images from a class-structured folder into a flat numpy matrix.

    Args:
        folder:  path containing one subfolder per class
        classes: list of class names (folder index = numeric label).
                 Defaults to SORTED_CLASSES (alphabetical).
        size:    target square pixel size for downsampling
        retries: retries per file on transient read errors
        verbose: print per-class counts

    Returns:
        X: (n, size*size*3) uint8 array of flattened pixels
        y: (n,) int array of class labels
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
    """Load train, val, and test splits in one call."""
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


def verify_data(data_dir=None):
    """Print per-class image counts for each split."""
    if data_dir is None:
        data_dir = DATA_DIR

    print(f'Data check: {os.path.abspath(data_dir)}\n')
    for split in ['processed', 'processed_val', 'processed_test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            print(f'{split}: MISSING ({split_path} does not exist)\n')
            continue
        print(f'{split}:')
        total = 0
        for cls in SORTED_CLASSES:
            cls_path = os.path.join(split_path, cls)
            n = len(glob.glob(os.path.join(cls_path, '*.jpg'))) if os.path.isdir(cls_path) else 0
            total += n
            print(f'  {cls:<26} {n:>6}')
        print(f'  {"total":<26} {total:>6}\n')


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    setup_data()