# FIFA FeatureScope — Feature Selection + Modeling Pipeline

## Project Overview

**Dataset:** FIFA 22 Players (`players_22.csv`) — 19,239 players, 110 raw columns  
**Goal:** Predict player position (14 classes: CB, ST, CM, CDM, LB, RB, CAM, RM, LM, RW, LW, RWB, LWB, CF)  
**Method:** Apply all 5 feature selection techniques in a progressive pipeline, then model with XGBoost and LightGBM

---

## Dataset After Cleaning

| Step | Action | Result |
|---|---|---|
| Load raw data | `low_memory=False` to handle mixed dtypes | 19,239 × 110 |
| Drop null rows | Rows with missing values removed | 16,020 × 110 |
| Drop junk columns | URLs, IDs, names, dates removed | 16,020 × 90 |
| Extract target | First position from `player_positions` → `position` | 16,020 × 81 |
| Drop string columns | Positional rating cols (`ls`, `st`, `rs`... etc) dropped | 16,020 × 58 |
| Encode categoricals | `preferred_foot`, `real_face` → 0/1; `work_rate` → LabelEncoder; `body_type` dropped | 16,020 × 57 |
| Final shape | Ready for feature selection | **16,020 × 56** (55 features + 1 target) |

---

## Step 1 — Variance Thresholding (Numerical Features)

### Concept
If a numerical feature has very low variance, it means the values barely change across observations. A near-constant column carries almost no information for distinguishing between players.

**Formula:**

$$\text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Important:** Always apply StandardScaler before VarianceThreshold on numerical features. Otherwise large-scale features (like `wage_eur` in millions) will always dominate over small-scale features (like `age`), making the threshold meaningless.

### Result
```
Threshold: 0 (drop only perfectly constant features)
Before: 55 features
After:  55 features
Dropped: 0
```

**Conclusion:** No feature in the FIFA dataset is perfectly constant — every attribute varies across 16,020 players. This is expected for a sports dataset designed with diverse player profiles.

---

## Step 2 — Variance Thresholding (Binary Features)

### Concept
Binary features (0/1 values) use the **Bernoulli variance formula** instead of the standard formula:

$$\text{Var}(X) = p(1 - p)$$

where `p` is the proportion of 1s in the column.

- Maximum variance = 0.25 at p = 0.5 (perfectly balanced)
- Variance approaches 0 as p approaches 0 or 1 (heavily imbalanced)

**Threshold logic:** If you want to drop features where one value appears more than 80% of the time, set threshold = `0.8 × (1 - 0.8) = 0.16`.

### Binary Features in Dataset
| Feature | p (proportion of 1s) | Variance | Decision |
|---|---|---|---|
| `preferred_foot` | 0.747 (74.7% right-footed) | 0.189 | ✅ Kept |
| `real_face` | 0.116 (11.6% have real face scan) | 0.102 | ❌ Dropped |

### Result
```
Threshold: 0.16  (drops features >80% dominated by one value)
Before: 2 binary features
After:  1 binary feature
Dropped: real_face
```

**Conclusion:** `real_face` is heavily imbalanced — only 11.6% of players have a real face scan. This column carries almost no discriminative information and is dropped.

---

## Step 3 — Handling Highly Correlated Features

### Concept
If two features are highly correlated, they carry almost the same information. Keeping both adds redundancy without adding signal, and can hurt some models (especially linear ones).

**Pearson Correlation:**

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

- r = 1.0 → perfect positive correlation
- r = -1.0 → perfect negative correlation
- r = 0.0 → no correlation

`abs(r)` is used to catch both positive and negative correlations.

### Method
1. Compute the full correlation matrix
2. Take the upper triangle (avoid checking pairs twice)
3. Drop any column that has `abs(r) > 0.85` with any other column

### Result
```
Threshold: 0.85
Before: 54 features
After:  36 features
Dropped: 18 features
```

**Dropped features:**
`release_clause_eur`, `attacking_finishing`, `attacking_short_passing`, `attacking_volleys`, `skill_dribbling`, `skill_ball_control`, `movement_acceleration`, `movement_sprint_speed`, `movement_reactions`, `power_shot_power`, `power_strength`, `power_long_shots`, `mentality_interceptions`, `mentality_positioning`, `mentality_vision`, `defending_marking_awareness`, `defending_standing_tackle`, `defending_sliding_tackle`

**Why these make sense:** Features like `movement_acceleration` and `movement_sprint_speed` are naturally correlated — fast players are fast in both metrics. Similarly, `defending_standing_tackle` and `defending_sliding_tackle` measure the same underlying defensive ability.

---

## Step 4 — SelectKBest with f_classif

### Concept
Unlike the previous methods, `f_classif` actually looks at the **target variable `y`**. It runs a one-way **ANOVA F-test** for each feature, measuring whether the feature's mean differs significantly across player position classes.

- **High F-score** → feature mean differs a lot across positions → feature separates classes well → keep
- **Low F-score** → feature mean is similar across all positions → feature doesn't help distinguish → drop

This is a **supervised, statistical** method — the first in the pipeline to use the target.

### Result
```
k = 20 (keep top 20 features by F-score)
Before: 36 features
After:  20 features
Dropped: 16 features
```

**Selected features:**

| Feature | Why it makes football sense |
|---|---|
| `height_cm` | GKs and CBs are significantly taller than wingers |
| `weight_kg` | Physical players (ST, CB) vs technical players (CAM, LM) |
| `preferred_foot` | Positional tendencies differ by foot |
| `skill_moves` | Wingers and CAMs have higher skill moves than CBs |
| `pace` | Wingers and fullbacks are faster than CDMs and CBs |
| `shooting` | Strikers and CAMs have high shooting, GKs/CBs don't |
| `passing` | CMs and CAMs have high passing |
| `dribbling` | Wingers and CAMs excel here |
| `defending` | CBs and CDMs have high defending |
| `physic` | CBs and CMs are more physical |
| `attacking_crossing` | Wingers and fullbacks cross more |
| `attacking_heading_accuracy` | CBs and STs win headers |
| `skill_curve` | Technical players like CAM and LM |
| `skill_fk_accuracy` | Specialist players |
| `skill_long_passing` | CDMs and CMs play long balls |
| `movement_agility` | Wingers are more agile |
| `movement_balance` | Technical players maintain balance better |
| `power_jumping` | CBs and STs jump higher |
| `mentality_aggression` | CDMs and CBs are more aggressive |
| `mentality_penalties` | Specific to attacking positions |

---

## Step 5 — RFECV (Recursive Feature Elimination with Cross Validation)

### Concept
RFECV is the most powerful method — it actually **trains a model** to decide which features matter, and uses **cross validation** to find the optimal number of features.

**Algorithm:**
1. Start with all features
2. Train a model and rank features by importance
3. Remove the least important feature
4. Retrain and cross-validate
5. Repeat until minimum features reached
6. Pick the feature count that gave best cross-validated accuracy

**Why cross validation?** Training and testing on the same data causes overfitting — the model memorizes training data and reports artificially high accuracy. CV tests on unseen data at each fold, giving a trustworthy performance estimate.

**Model used:** `LinearSVC` — faster than Logistic Regression for 14-class classification with 16,020 samples. Features scaled with StandardScaler before fitting.

### Result
```
cv = 5 folds
Before: 20 features
After:  20 features
Dropped: 0
```

**Conclusion:** RFECV confirmed that all 20 features selected by SelectKBest are genuinely useful. Removing any of them degrades cross-validated accuracy — so all 20 are kept as the final optimal feature set.

---

## Final Pipeline Summary

| Step | Method | Looks at y? | Uses Model? | Features Before | Features After | Dropped |
|---|---|---|---|---|---|---|
| 1 | Variance Threshold (Numerical) | ❌ | ❌ | 55 | 55 | 0 |
| 2 | Variance Threshold (Binary) | ❌ | ❌ | 55 | 54 | 1 |
| 3 | Correlation Handling | ❌ | ❌ | 54 | 36 | 18 |
| 4 | SelectKBest (f_classif) | ✅ | ❌ | 36 | 20 | 16 |
| 5 | RFECV (LinearSVC) | ✅ | ✅ | 20 | 20 | 0 |

**Final feature set: 20 features** (from original 55 — 63.6% reduction)

---

## Final 20 Features

```python
['height_cm', 'weight_kg', 'preferred_foot', 'skill_moves', 'pace',
 'shooting', 'passing', 'dribbling', 'defending', 'physic',
 'attacking_crossing', 'attacking_heading_accuracy', 'skill_curve',
 'skill_fk_accuracy', 'skill_long_passing', 'movement_agility',
 'movement_balance', 'power_jumping', 'mentality_aggression',
 'mentality_penalties']
```

---

---

# Modeling Phase — XGBoost & LightGBM

## Setup

The 20 selected features were saved to `fifa_model_ready.csv` with the target column `position` (string labels). In the modeling notebook, `LabelEncoder` was applied to encode positions as integers before training.

**Train/Test Split:**
```
Strategy: Stratified (stratify=y)
Reason: 14 position classes with heavy imbalance — stratification ensures
        proportional class representation in both train and test sets
Test size: 20%
X_train: (12816, 20) | X_test: (3204, 20)
```

---

## Class Imbalance Problem

After the first XGBoost baseline run, three positions were completely ignored:

| Position | Support | F1 Score |
|---|---|---|
| CF (Centre Forward) | 26 | 0.00 |
| LWB (Left Wing Back) | 33 | 0.00 |
| RWB (Right Wing Back) | 34 | 0.00 |

**Root cause:** These are rare positions in real football — very few players occupy them, so the dataset naturally has very few samples. XGBoost ignores them because predicting majority classes gives better overall accuracy.

**Why Macro F1 is the correct metric here:** Accuracy is dominated by large classes (CB: 628, ST: 472). Macro F1 weights every class equally regardless of size, making it the honest measure for this imbalanced multiclass problem.

---

## Experiment 1 — XGBoost Baseline

```python
XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
              eval_metric='mlogloss', random_state=42)
```

| Metric | Score |
|---|---|
| Accuracy | 0.688 |
| Macro F1 | 0.44 |

CF, LWB, RWB → all 0.00 F1. Model completely ignores minority classes.

---

## Experiment 2 — XGBoost + SMOTE

SMOTE (Synthetic Minority Oversampling Technique) applied **only on training data** to avoid data leakage. Synthetic samples generated for minority classes to balance the distribution.

**Critical rule:** SMOTE must be applied after the train/test split, never before. Applying it before would let synthetic samples derived from the test distribution bleed into training, making evaluation unreliable.

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

| Metric | Score |
|---|---|
| Accuracy | 0.651 |
| Macro F1 | 0.45 |

Accuracy dropped but Macro F1 improved — CF, LWB, RWB now get non-zero scores. This is the correct trade-off: the model became more fair across all classes even though raw accuracy fell.

---

## Experiment 3 — XGBoost + SMOTE + RandomizedSearchCV

```python
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}
# Best params found: n_estimators=200, max_depth=8, learning_rate=0.1
```

| Metric | Score |
|---|---|
| Accuracy | 0.668 |
| Macro F1 | 0.45 |

Tuning recovered some accuracy but Macro F1 stayed flat. The bottleneck is not the model or hyperparameters — it's the data. CF, LWB, RWB simply don't have enough samples for tuning to fix.

---

## Experiment 4 — LightGBM + class_weight='balanced'

Instead of SMOTE, LightGBM's built-in `class_weight='balanced'` was used. This tells the model to internally assign higher loss penalties to minority class errors — different mechanism, same goal as SMOTE.

```python
LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
# Best params: n_estimators=300, max_depth=4, learning_rate=0.05
```

| Metric | Score |
|---|---|
| Accuracy | 0.664 |
| Macro F1 | **0.46** |

Best Macro F1 so far. RWB improved from 0.00 to 0.15.

---

## Experiment 5 — LightGBM + SMOTE (Leaky — Incorrect)

SMOTE applied outside CV loop, then LightGBM trained on resampled data with `RandomizedSearchCV`. This produced a misleadingly high CV score:

```
CV Macro F1:   0.914  ← inflated (leaky)
Test Macro F1: 0.45   ← real performance
```

**Why this happened:** SMOTE creates synthetic samples very similar to existing ones. When CV runs on pre-resampled data, validation folds contain synthetic samples nearly identical to training samples. The model appears to generalize well but it's actually memorizing the synthetic data. This is a data leakage problem.

---

## Experiment 6 — LightGBM + SMOTE Pipeline (Correct)

The correct approach: SMOTE applied **inside each CV fold** using `imblearn.Pipeline`. This ensures synthetic samples never appear in validation folds.

```python
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LGBMClassifier(random_state=42, verbose=-1))
])

param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [4, 6, 8]
}
# Best params: n_estimators=100, max_depth=8, learning_rate=0.05
```

Now CV score and test score are consistent — no leakage:

```
CV Macro F1:   0.458  ← honest
Test Macro F1: 0.46   ← real performance
```

| Metric | Score |
|---|---|
| Accuracy | 0.663 |
| Macro F1 | **0.46** |

---

## Final Model Comparison

| Run | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| XGBoost Baseline | 0.688 | 0.44 | CF/LWB/RWB ignored |
| XGBoost + SMOTE | 0.651 | 0.45 | Minority classes recognized |
| XGBoost + SMOTE + Tuning | 0.668 | 0.45 | Marginal gain from tuning |
| LightGBM + class_weight | 0.664 | **0.46** | Best — clean approach |
| LightGBM + SMOTE (leaky) | 0.667 | 0.45 | Inflated CV — not trustworthy |
| LightGBM + SMOTE Pipeline | 0.663 | **0.46** | Best — correct approach |

**Winner:** LightGBM with either `class_weight='balanced'` or the correct SMOTE Pipeline — both achieve Macro F1 of 0.46 with honest evaluation.

---

## Confusion Matrix Observations

The confusion matrix revealed that model errors are not random — they follow positional logic:

- **LWB ↔ LB** — most LWB players misclassified as LB (very similar roles)
- **RWB ↔ RB** — same pattern on the right side
- **LW / LM / RM / RW** — frequent confusion among each other (tactically interchangeable)
- **CB, ST, GK** — very clean predictions (distinctive profiles)

This is a **domain knowledge problem** as much as a modeling problem. Even human scouts sometimes debate whether a player is a LM or LW.

---

## Key Learnings from Modeling Phase

1. **Accuracy is misleading for imbalanced multiclass problems** — always use Macro F1 as primary metric when classes have very different sizes.

2. **SMOTE must be applied inside the CV loop** — applying it before CV creates data leakage, inflating CV scores dramatically (0.91 vs real 0.45 in this experiment). Use `imblearn.Pipeline` to handle this correctly.

3. **There is a data ceiling** — all approaches plateau around 0.46 Macro F1. Beyond a certain point, no algorithm or technique can compensate for genuinely insufficient data in minority classes.

4. **`class_weight='balanced'` vs SMOTE** — both solve the same imbalance problem through different mechanisms. `class_weight` adjusts loss weights internally; SMOTE creates synthetic samples externally. Performance is similar when both are applied correctly.

5. **XGBoost vs LightGBM** — LightGBM edges out XGBoost slightly (0.46 vs 0.45) on this dataset, and is significantly faster to train. For tabular classification, LightGBM is generally preferred in production.

6. **RandomizedSearchCV over GridSearchCV** — for large param spaces with expensive models, RandomizedSearchCV samples a random subset of combinations. Much faster, usually finds equally good or better results.

7. **Confusion matrix tells more than accuracy** — the confusion matrix revealed systematic positional confusion (LWB/LB, RWB/RB) that accuracy alone would never surface.

---

## Key Learnings from Feature Selection Phase

1. **Not all methods will always drop features** — VarianceThreshold dropped nothing on numerical features because FIFA data is naturally varied. That is a valid result, not a failure.

2. **Method order matters** — cheap unsupervised methods first (VarianceThreshold, Correlation), expensive supervised methods last (SelectKBest, RFECV).

3. **Scale before model-based methods** — StandardScaler is essential before RFECV with LinearSVC to ensure fair feature comparison.

4. **chi2 vs f_classif** — chi2 is for non-negative integer/count features; f_classif is for continuous numerical features. FIFA attributes are continuous so f_classif is the correct choice here.

5. **RFECV confirming SelectKBest** — when RFECV keeps all features that SelectKBest chose, it validates that the statistical test and the model-based test agree. Strong signal that the feature set is genuinely optimal.
