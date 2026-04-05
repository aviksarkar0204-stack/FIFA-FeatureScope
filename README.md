# ⚽ FIFA FeatureScope

A feature selection practice project built on the FIFA 22 player dataset. The pipeline demonstrates all 5 core feature selection techniques to predict player position from player attributes.

---

## 📊 Dataset

- **Source:** [FIFA 22 Complete Player Dataset — Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)
- **File used:** `players_22.csv`
- **Raw shape:** 19,239 players × 110 columns
- **After cleaning:** 16,020 players × 55 features
- **Target:** Player position (14 classes — CB, ST, CM, CDM, LB, RB, CAM, RM, LM, RW, LW, RWB, LWB, CF)

---

## 🔧 Feature Selection Pipeline

| Step | Method | Features Before | Features After | Dropped |
|---|---|---|---|---|
| 1 | Variance Threshold (Numerical) | 55 | 55 | 0 |
| 2 | Variance Threshold (Binary) | 55 | 54 | 1 |
| 3 | Correlation Handling | 54 | 36 | 18 |
| 4 | SelectKBest (f_classif) | 36 | 20 | 16 |
| 5 | RFECV (LinearSVC) | 20 | 20 | 0 |

**Final feature set: 20 features** (63.6% reduction from original 55)

---

## 🏆 Final 20 Features

```
height_cm, weight_kg, preferred_foot, skill_moves, pace,
shooting, passing, dribbling, defending, physic,
attacking_crossing, attacking_heading_accuracy, skill_curve,
skill_fk_accuracy, skill_long_passing, movement_agility,
movement_balance, power_jumping, mentality_aggression, mentality_penalties
```

---

## 🚀 Streamlit App

> 🔧 **Coming soon** — 5-page interactive app, one page per feature selection method.

<!-- 
Planned pages:
- Page 1: Variance Threshold (Numerical)
- Page 2: Variance Threshold (Binary)
- Page 3: Correlation Heatmap
- Page 4: SelectKBest scores and rankings
- Page 5: RFECV optimal feature curve
-->

---

## 🛠️ Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- Streamlit *(coming soon)*

---

## 📁 Project Structure

```
FIFA-FeatureScope/
├── players_22.csv               # Raw dataset (not tracked)
├── players_22_clean.csv         # Cleaned dataset
├── data_cleaning.ipynb          # Cleaning notebook
├── feature_selection.ipynb      # Full pipeline notebook
├── FIFA_FeatureScope_Pipeline.md # Detailed pipeline explanation
└── README.md
```

---

## 👤 Author

**Avik Sarkar**  
B.Tech CSE (AI/ML), Brainware University  
[GitHub](https://github.com/aviksarkar0204-stack) • [Hugging Face](https://huggingface.co/Avik128)
