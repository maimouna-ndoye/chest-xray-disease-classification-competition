# 🫁 Chest X-Ray Disease Classification Competition

> **Deep Learning Competition** | Medical Imaging | Macro F1-Score

---

## 📌 Overview

Welcome to the **Chest X-Ray Disease Classification Competition**!

Your goal is to build the best deep learning model that can automatically classify chest X-ray images into pulmonary disease categories.

This competition simulates a real clinical problem: helping doctors detect lung diseases faster and more accurately from radiographic images.

**The final ranking will be based on the Macro F1-Score computed on a hidden test set.**

---

## 🏷️ Classes

The dataset contains **3 classes**:

| Label | Class | Description |
|-------|-------|-------------|
| `0` | `NORMAL` | Healthy lungs, no disease detected |
| `1` | `BACTERIA` | Signs of bacterial pneumonia |
| `2` | `VIRUS` | Signs of viral pneumonia |

---

## 📦 Dataset

The dataset is derived from the **Chest X-Ray Pneumonia** dataset, reorganized into a 3-class classification task.

### ⬇️ Download the Dataset

> **[👉 Click here to download the dataset from Google Drive](https://drive.google.com/drive/folders/1f8eGm8I5zDIm98Ove-Rbhv_bXCF5i7rE?usp=drive_link)**

### Dataset Structure

```
dataset/
├── train/               ← Use this to train your model
│   ├── NORMAL/
│   ├── BACTERIA/
│   └── VIRUS/
│
├── val/                 ← Use this to tune your model
│   ├── NORMAL/
│   ├── BACTERIA/
│   └── VIRUS/
│
└── test_images/         ← Generate predictions on these (unlabeled)
```

> ⚠️ `test_images/` has no labels. The hidden labels are kept by the organizers for final scoring.

---

## 📊 Evaluation Metric

Submissions are evaluated using **Macro F1-Score**, which treats all 3 classes equally — important given the class imbalance in medical datasets.

---

## 📤 Submission Format

Submit a file named **`predictions.txt`** with **one predicted label per line**, in the **same order as the test images sorted alphabetically**.

```
0
2
1
1
0
```

Where: `0` = NORMAL, `1` = BACTERIA, `2` = VIRUS

> ⚠️ Always sort your test images before predicting:
> ```python
> import os
> test_images = sorted(os.listdir("dataset/test_images/"))
> ```

---

## 🚀 Quick Start (Google Colab)

```python
# 1. Mount your Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Set your paths
TRAIN_DIR = '/content/drive/MyDrive/dataset/train'
VAL_DIR   = '/content/drive/MyDrive/dataset/val'
TEST_DIR  = '/content/drive/MyDrive/dataset/test_images'

# 3. Enable GPU: Runtime → Change runtime type → T4 GPU
```

A full **baseline notebook** is available in `competition_tools/`.

---

## 📋 Rules

1. ✅ Any framework allowed (TensorFlow, PyTorch, etc.)
2. ✅ Pre-trained models allowed (ResNet, EfficientNet, DenseNet, etc.)
3. ✅ Data augmentation allowed
4. ✅ Maximum **3 submissions** per team
5. ❌ External datasets **not allowed**
6. ❌ Code sharing between teams **not allowed**
7. ✅ Final ranking based **only on the hidden test set**

---

## 📅 Timeline

| Event | Date |
|-------|------|
| Competition opens | TBD |
| Submission deadline | TBD |
| Final results revealed | TBD |

---

## 🏆 Leaderboard

Scores will be updated in the `leaderboard/` folder after each submission round.

| Rank | Team | Macro F1-Score |
|------|------|---------------|
| — | Baseline (provided) | ~0.75 |

---

## 📂 Repository Structure

```
chest-xray-disease-classification-competition/
├── README.md                  ← This file
├── competition_tools/         ← Baseline notebook & evaluation script
├── dataset/                   ← Dataset info (download from Drive above)
├── leaderboard/               ← Rankings updated here
├── submission_format/         ← Example of valid submission
└── .gitignore
```

---

## 👥 Organizers

| Name | Role |
|------|------|
| Maimouna Ndoye | Dataset preparation & splitting |
| [Your Name] | Competition setup & baseline |

---



---

*Good luck to all teams! 🏥🤖*
