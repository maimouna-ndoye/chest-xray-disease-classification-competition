"""
=============================================================
  Chest X-Ray Competition — Dataset Re-Splitting Script
=============================================================

WHAT THIS DOES:
  Pools all images from the original Kaggle chest X-ray dataset,
  re-splits them into a harder 3-class competition format:
    - Class 0: NORMAL
    - Class 1: BACTERIAL PNEUMONIA
    - Class 2: VIRAL PNEUMONIA

  Output structure:
    competition_data/
      train/
        NORMAL/         (~700 images  — 20% of total train)
        BACTERIAL/      (~1,900 images — 55% of total train)
        VIRAL/          (~900 images  — 25% of total train)
      val/
        NORMAL/         (50 images — balanced)
        BACTERIAL/      (50 images — balanced)
        VIRAL/          (50 images — balanced)
      test/
        NORMAL/         (~233 images — balanced, labels hidden from teams)
        BACTERIAL/      (~233 images)
        VIRAL/          (~233 images)
      test_images/      (flat folder of test images, no subfolders — what teams receive)
        0.jpeg, 1.jpeg, ...
      test_labels.npy   (ground truth — YOU keep this, don't upload)
      test_filemap.csv  (maps filename → original path, for your reference)
      class_info.txt    (competition instructions snippet)

SETUP:
  pip install numpy pillow scikit-learn tqdm pandas

HOW TO RUN:
  1. Download dataset from Kaggle:
     https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  2. Unzip so you have: chest_xray/train/, chest_xray/test/, chest_xray/val/
  3. Set KAGGLE_DATA_DIR below to point to that folder
  4. Run: python prepare_competition_data.py
=============================================================
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG — edit this path
# ─────────────────────────────────────────────
KAGGLE_DATA_DIR = "./chest_xray"       # folder containing train/ test/ val/
OUTPUT_DIR      = "./competition_data"
RANDOM_SEED     = 2025                 # keep this secret — don't publish it

# Split sizes
TRAIN_SIZE      = 0.70   # 70% of all images → training
VAL_SIZE        = 0.025  # ~150 images → validation (capped, see below)
TEST_SIZE       = 0.275  # remaining → test (labels hidden)

VAL_PER_CLASS   = 50     # exactly 50 per class in val (small → harder to tune)
TEST_PER_CLASS  = 233    # ~233 per class in test (balanced)

# Imbalance ratios for TRAIN set (makes it harder — different from original)
TRAIN_RATIOS = {
    "NORMAL":    0.20,   # 20% — underrepresented
    "BACTERIAL": 0.55,   # 55% — dominant class
    "VIRAL":     0.25,   # 25% — minority
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# STEP 1: COLLECT ALL IMAGES & ASSIGN 3-CLASS LABELS
# ─────────────────────────────────────────────
print("=" * 60)
print("  Chest X-Ray Competition Data Preparation")
print("=" * 60)
print(f"\n[1/5] Scanning {KAGGLE_DATA_DIR} for all images...")

all_images = []   # list of (filepath, label_str)

splits = ["train", "test", "val"]
for split in splits:
    split_dir = Path(KAGGLE_DATA_DIR) / split

    if not split_dir.exists():
        print(f"  ⚠️  Skipping missing folder: {split_dir}")
        continue

    # NORMAL images
    normal_dir = split_dir / "NORMAL"
    if normal_dir.exists():
        for f in normal_dir.iterdir():
            if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                all_images.append((str(f), "NORMAL"))

    # PNEUMONIA images — split into BACTERIAL and VIRAL by filename
    pneumonia_dir = split_dir / "PNEUMONIA"
    if pneumonia_dir.exists():
        for f in pneumonia_dir.iterdir():
            if f.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            name = f.name.lower()
            if "bacteria" in name:
                all_images.append((str(f), "BACTERIAL"))
            elif "virus" in name:
                all_images.append((str(f), "VIRAL"))
            else:
                # fallback: treat unknown pneumonia as bacterial (most common)
                all_images.append((str(f), "BACTERIAL"))

random.shuffle(all_images)

# Count per class
label_counts = Counter(label for _, label in all_images)
print(f"\n  Total images found: {len(all_images)}")
for cls, count in sorted(label_counts.items()):
    print(f"    {cls:<12}: {count:,} images")

# ─────────────────────────────────────────────
# STEP 2: SEPARATE BY CLASS
# ─────────────────────────────────────────────
print("\n[2/5] Separating by class...")

by_class = {"NORMAL": [], "BACTERIAL": [], "VIRAL": []}
for path, label in all_images:
    by_class[label].append(path)

for cls in by_class:
    random.shuffle(by_class[cls])

# ─────────────────────────────────────────────
# STEP 3: ASSIGN VAL AND TEST FIRST (balanced)
# ─────────────────────────────────────────────
print(f"\n[3/5] Creating balanced val ({VAL_PER_CLASS}/class) "
      f"and test ({TEST_PER_CLASS}/class) sets...")

val_set   = {}   # class → list of paths
test_set  = {}
train_set = {}

for cls, paths in by_class.items():
    needed = VAL_PER_CLASS + TEST_PER_CLASS
    if len(paths) < needed:
        raise ValueError(
            f"Not enough images for {cls}: have {len(paths)}, need {needed}. "
            f"Reduce VAL_PER_CLASS or TEST_PER_CLASS."
        )
    val_set[cls]   = paths[:VAL_PER_CLASS]
    test_set[cls]  = paths[VAL_PER_CLASS : VAL_PER_CLASS + TEST_PER_CLASS]
    train_set[cls] = paths[VAL_PER_CLASS + TEST_PER_CLASS:]

# ─────────────────────────────────────────────
# STEP 4: APPLY IMBALANCE TO TRAINING SET
# ─────────────────────────────────────────────
print("\n[4/5] Applying imbalanced ratios to training set...")

# Find the limiting class (smallest pool) and scale train to that
min_ratio    = min(TRAIN_RATIOS.values())
min_cls      = min(TRAIN_RATIOS, key=TRAIN_RATIOS.get)
max_possible = int(len(train_set[min_cls]) / min_ratio)

final_train = {}
for cls, ratio in TRAIN_RATIOS.items():
    n = int(max_possible * ratio)
    available = len(train_set[cls])
    if n > available:
        print(f"  ⚠️  {cls}: requested {n} but only {available} available — using all")
        n = available
    final_train[cls] = train_set[cls][:n]

print("\n  Final dataset sizes:")
print(f"  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
print(f"  {'-'*40}")
total_train = 0
for cls in ["NORMAL", "BACTERIAL", "VIRAL"]:
    t = len(final_train[cls])
    v = len(val_set[cls])
    te = len(test_set[cls])
    total_train += t
    print(f"  {cls:<12} {t:>8,} {v:>8,} {te:>8,}")
print(f"  {'TOTAL':<12} {total_train:>8,} "
      f"{sum(len(v) for v in val_set.values()):>8,} "
      f"{sum(len(v) for v in test_set.values()):>8,}")

# ─────────────────────────────────────────────
# STEP 5: COPY FILES INTO OUTPUT STRUCTURE
# ─────────────────────────────────────────────
print(f"\n[5/5] Writing competition data to {OUTPUT_DIR}/ ...")

CLASS_TO_ID = {"NORMAL": 0, "BACTERIAL": 1, "VIRAL": 2}
out = Path(OUTPUT_DIR)

def copy_to(paths, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in tqdm(paths, desc=f"  → {dest_dir.relative_to(out)}", leave=False):
        shutil.copy2(src, dest_dir / Path(src).name)

# Train
for cls, paths in final_train.items():
    copy_to(paths, out / "train" / cls)

# Val
for cls, paths in val_set.items():
    copy_to(paths, out / "val" / cls)

# Test — labelled folders (for YOUR reference)
for cls, paths in test_set.items():
    copy_to(paths, out / "test" / cls)

# Test — flat folder for teams (no subfolder labels)
print("\n  Creating flat test_images/ folder for teams...")
flat_dir = out / "test_images"
flat_dir.mkdir(parents=True, exist_ok=True)

test_records = []   # (new_filename, original_path, class, label_id)

idx = 0
for cls in ["NORMAL", "BACTERIAL", "VIRAL"]:
    for src in test_set[cls]:
        suffix   = Path(src).suffix
        new_name = f"{idx}{suffix}"
        shutil.copy2(src, flat_dir / new_name)
        test_records.append({
            "filename":  new_name,
            "original":  src,
            "class":     cls,
            "label_id":  CLASS_TO_ID[cls]
        })
        idx += 1

# Shuffle so label order isn't guessable from filename
random.shuffle(test_records)
# Rename after shuffle to keep 0, 1, 2, ... ordering
for i, rec in enumerate(test_records):
    old_path = flat_dir / rec["filename"]
    suffix   = Path(rec["filename"]).suffix
    new_name = f"{i}{suffix}"
    old_path.rename(flat_dir / new_name)
    rec["filename"] = new_name

# Save test labels (YOU keep this — do NOT upload)
labels_array = np.array([r["label_id"] for r in test_records])
np.save(str(out / "test_labels.npy"), labels_array)

# Save filemap CSV (for your reference)
df = pd.DataFrame(test_records)
df.to_csv(str(out / "test_filemap.csv"), index=False)

print(f"  ✅ {len(test_records)} test images written to test_images/")
print(f"  ✅ test_labels.npy saved  ← KEEP PRIVATE, do not upload")
print(f"  ✅ test_filemap.csv saved ← for your reference")

# ─────────────────────────────────────────────
# WRITE CLASS INFO FILE (for README / competition page)
# ─────────────────────────────────────────────
class_info = f"""
COMPETITION DATASET — CHEST X-RAY DISEASE CLASSIFICATION
=========================================================

TASK: 3-class image classification
  Class 0 → NORMAL
  Class 1 → BACTERIAL PNEUMONIA
  Class 2 → VIRAL PNEUMONIA

DATASET SPLIT:
  train/        — {total_train:,} images (class-imbalanced, see below)
  val/           — {VAL_PER_CLASS * 3} images (50 per class, balanced)
  test_images/   — {len(test_records)} images (balanced, labels hidden)

TRAINING CLASS DISTRIBUTION:
  NORMAL     : {len(final_train['NORMAL']):,} images ({len(final_train['NORMAL'])/total_train*100:.1f}%)
  BACTERIAL  : {len(final_train['BACTERIAL']):,} images ({len(final_train['BACTERIAL'])/total_train*100:.1f}%)
  VIRAL      : {len(final_train['VIRAL']):,} images ({len(final_train['VIRAL'])/total_train*100:.1f}%)

EVALUATION METRIC: Macro F1-Score
  (average F1 across all 3 classes — rewards handling class imbalance)

SUBMISSION FORMAT:
  A text file (predictions.txt) with one integer per line (0, 1, or 2),
  ordered by image index (0.jpeg, 1.jpeg, ...).

EXAMPLE predictions.txt:
  0
  2
  1
  ...
"""

with open(str(out / "class_info.txt"), "w") as f:
    f.write(class_info)

print(class_info)

# ─────────────────────────────────────────────
# WHAT TO UPLOAD vs KEEP PRIVATE
# ─────────────────────────────────────────────
print("=" * 60)
print("  UPLOAD TO KAGGLE / GITHUB:")
print("    ✅ competition_data/train/")
print("    ✅ competition_data/val/")
print("    ✅ competition_data/test_images/   (flat, no labels)")
print("    ✅ competition_data/class_info.txt")
print()
print("  KEEP PRIVATE (never upload):")
print("    ❌ competition_data/test/          (labelled test folders)")
print("    ❌ competition_data/test_labels.npy")
print("    ❌ competition_data/test_filemap.csv")
print("    ❌ RANDOM_SEED value (2025)")
print("=" * 60)
print("\n✅ Done! Your competition dataset is ready.")
