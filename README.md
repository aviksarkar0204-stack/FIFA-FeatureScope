# ⚽ FIFA FeatureScope

A feature selection + modeling practice project built on the FIFA 22 player dataset. The pipeline demonstrates all 5 core feature selection techniques, followed by XGBoost and LightGBM classification to predict player position from player attributes.

> 📄 For full pipeline and modeling details, see [FIFA_FeatureScope_Pipeline.md](FIFA_FeatureScope_Pipeline.md)

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

## 🤖 Modeling Phase

The 20 selected features were used to train and compare XGBoost and LightGBM classifiers. A stratified 80/20 train/test split was used due to the multiclass imbalance problem.

**Primary metric: Macro F1** (not accuracy) — because classes like CF (26 samples) and RWB (34 samples) are heavily underrepresented compared to CB (628) and ST (472). Accuracy is dominated by majority classes and is misleading here.

### Experiments & Results

| Run | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| XGBoost Baseline | 0.688 | 0.44 | CF / LWB / RWB completely ignored |
| XGBoost + SMOTE | 0.651 | 0.45 | Minority classes recognized |
| XGBoost + SMOTE + Tuning | 0.668 | 0.45 | Marginal gain from RandomizedSearchCV |
| LightGBM + class_weight | 0.664 | **0.46** | Best — clean approach |
| LightGBM + SMOTE (leaky) | 0.667 | 0.45 | CV inflated to 0.91 — not trustworthy |
| LightGBM + SMOTE Pipeline | 0.663 | **0.46** | Best — correct approach |

### Key Findings

- **LightGBM** slightly outperforms XGBoost on this dataset (0.46 vs 0.45 Macro F1)
- **SMOTE must be applied inside the CV loop** using `imblearn.Pipeline` — applying it before CV causes data leakage, inflating CV scores from 0.46 to 0.91
- **`class_weight='balanced'`** and **SMOTE Pipeline** achieve the same result through different mechanisms — both correct, both honest
- **Data ceiling reached at 0.46 Macro F1** — CF, LWB, RWB don't have enough samples for any technique to fully fix
- Confusion matrix showed systematic positional confusion: LWB↔LB, RWB↔RB, LW↔LM↔RM↔RW — tactically similar positions are naturally hard to distinguish

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
- imbalanced-learn (SMOTE, Pipeline)
- xgboost, lightgbm
- seaborn, matplotlib
- Streamlit *(coming soon)*

---

## 📁 Project Structure

```
FIFA-FeatureScope/
├── players_22.csv                # Raw dataset (not tracked)
├── players_22_clean.csv          # Cleaned dataset
├── fifa_model_ready.csv          # Final 20 features + target (model input)
├── data_cleaning.ipynb           # Cleaning + feature selection notebook
├── Gradient_Boosting.ipynb       # XGBoost & LightGBM modeling notebook
├── FIFA_FeatureScope_Pipeline.md # Detailed pipeline + modeling explanation
└── README.md
```

---

## 👤 Author

**Avik Sarkar**  
B.Tech CSE (AI/ML), Brainware University  
[GitHub](https://github.com/aviksarkar0204-stack) • [Hugging Face](https://huggingface.co/Avik128)
