# 🏆 Leaderboard — Chest X-Ray Disease Classification Competition

> **Metric:** Macro F1-Score | **Test images:** 352 | **Last updated:** March 2026

---

## 🥇 Final Rankings

| Rank | Team | NORMAL (F1) | BACTERIAL (F1) | VIRAL (F1) | **Macro F1** |
|------|------|:-----------:|:--------------:|:----------:|:------------:|
| 🥇 1st | KMF | 0.1757 | 0.4314 | 0.3840 | **0.3303** |
| 🥈 2nd | Julia&Adama | 0.1419 | 0.4365 | 0.4050 | **0.3278** |
| 🥉 3rd | Lieumo&ANsoumane&Daouda | 0.3169 | 0.3581 | 0.2667 | **0.3139** |

---

## 📊 Score Details

### 🥇 KMF — Macro F1: 0.3303
- Best overall score 🏆
- Good performance on **BACTERIAL** (0.4314) and **VIRAL** (0.3840)
- Struggled with **NORMAL** class (0.1757)

### 🥈 Julia — Macro F1: 0.3278
- Best at detecting **VIRAL** (0.4050) among all teams
- Struggled with **NORMAL** class (0.1419)

### 🥉 Lieumo — Macro F1: 0.3139
- Most **balanced** across all 3 classes
- Best at detecting **NORMAL** (0.3169) among all participants

---

## 📈 Score Comparison

```
NORMAL      : KMF  0.1757 | Julia 0.1419 | Lieumo 0.3169
BACTERIAL   : KMF  0.4314 | Julia 0.4365 | Lieumo 0.3581
VIRAL       : KMF  0.3840 | Julia 0.4050 | Lieumo 0.2667
─────────────────────────────────────────────────────────
MACRO F1    : KMF  0.3303 | Julia 0.3278 | Lieumo 0.3139
```

---

## ℹ️ Scoring Info

- Scores computed on **352 hidden test images**
- Metric: **Macro F1-Score** (equal weight for all 3 classes)
- Classes: `0 = NORMAL` | `1 = BACTERIAL` | `2 = VIRAL`

> 💡 **Note:** Scores around 0.33 reflect the difficulty of the dataset.
> The training set is intentionally imbalanced (BACTERIAL = 50%, NORMAL = 26%, VIRAL = 24%),
> making NORMAL the hardest class to detect. A perfect model would score **1.0**.

---

*Competition organized by Maimouna Ndoye & [Your Name] — Deep Learning Course 2026*
