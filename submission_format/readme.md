# Submission Format

This document explains how teams must submit their predictions for the **Chest X-Ray Disease Classification Competition**.

## 1. Submission File

Teams must submit a file named:

```
predictions.txt
```

The file must contain **one prediction per line**, corresponding to each image in the `dataset/test_images/` folder.

Example:

```
0
2
1
1
0
2
```

Each line represents the predicted class for one image.

---

## 2. Class Labels

Use the following numeric labels:

| ID | Class Name          |
| -- | ------------------- |
| 0  | NORMAL              |
| 1  | BACTERIAL PNEUMONIA |
| 2  | VIRAL PNEUMONIA     |

Example interpretation:

```
0 → NORMAL
1 → BACTERIAL PNEUMONIA
2 → VIRAL PNEUMONIA
```

---

## 3. Order of Predictions

Predictions **must follow the exact order of images** in:

```
dataset/test_images/
```

For example:

```
dataset/test_images/
0.jpeg
1.jpeg
2.jpeg
3.jpeg
...
```

The first line in `predictions.txt` corresponds to `0.jpeg`,
the second line corresponds to `1.jpeg`, etc.

---

## 4. Number of Predictions

The submission file must contain **exactly the same number of lines as the number of test images**.

Example:

* If there are **699 test images**
* The file must contain **699 lines**

---

## 5. File Naming Convention

Teams should submit their predictions using the following format:

```
TeamName_predictions.txt
```

Examples:

```
AI_team_predictions.txt
DeepVision_predictions.txt
MedAI_predictions.txt
```

---

## 6. How Submissions Are Evaluated

Submissions will be evaluated using **Macro F1 Score**.

The competition organizers will run the evaluation script:

```
python score.py TeamName_predictions.txt
```

The script will compute:

* Precision
* Recall
* F1-score for each class
* Final **Macro F1 Score**

Teams will be ranked on the leaderboard based on this score.

---

## 7. Example Submission

Example `predictions.txt`:

```
0
1
2
1
0
2
1
0
```

Make sure that:

* Only values **0, 1, or 2** are used
* The file contains **no extra spaces or text**
* Each prediction is on **a new line**
