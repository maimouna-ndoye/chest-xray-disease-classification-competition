# Chest X-Ray Disease Classification Competition

## Overview

This competition challenges participants to build a deep learning model that can classify chest X-ray images into different pulmonary disease categories.

Participants will train their models using the provided dataset and submit predictions for unseen test images.

The final ranking will be based on the Macro F1 Score computed on a hidden test set.

---

## Classes

The dataset contains three classes:

0 — NORMAL
1 — BACTERIAL PNEUMONIA
2 — VIRAL PNEUMONIA

---

## Dataset

The dataset is derived from the **Chest X-Ray Pneumonia dataset** and has been reorganized into a 3-class classification task.

Dataset structure:

dataset/train/ → training images
dataset/val/ → validation images
dataset/test_images/ → unlabeled test images for competition

---

## Task

Build a model that predicts the correct class for each chest X-ray image.

---

## Evaluation Metric

Submissions are evaluated using **Macro F1 Score**, which gives equal importance to all classes.

---

## Submission Format

Participants must submit a file named:

predictions.txt

The file should contain one integer per line corresponding to the predicted class for each test image.

Example:

0
2
1
1
0

---

## Competition Workflow

1. Train your model using the training dataset.
2. Validate your model using the validation dataset.
3. Generate predictions for the test images.
4. Submit your predictions.txt file.

The competition organizers will compute the final score using a private test label set.

---

## Tools

Participants may use any deep learning framework such as PyTorch or TensorFlow.

---

## Goal

The goal of this competition is to explore deep learning techniques for medical image classification and improve model robustness on imbalanced datasets.
