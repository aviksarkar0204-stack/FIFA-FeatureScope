# FIFA FeatureScope — Feature Selection Pipeline

## Project Overview

**Dataset:** FIFA 22 Players (`players_22.csv`) — 19,239 players, 110 raw columns  
**Goal:** Predict player position (14 classes: CB, ST, CM, CDM, LB, RB, CAM, RM, LM, RW, LW, RWB, LWB, CF)  
**Method:** Apply all 5 feature selection techniques in a progressive pipeline

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

## Key Learnings

1. **Not all methods will always drop features** — VarianceThreshold dropped nothing on numerical features because FIFA data is naturally varied. That is a valid result, not a failure.

2. **Method order matters** — cheap unsupervised methods first (VarianceThreshold, Correlation), expensive supervised methods last (SelectKBest, RFECV).

3. **Scale before model-based methods** — StandardScaler is essential before RFECV with LinearSVC to ensure fair feature comparison.

4. **chi2 vs f_classif** — chi2 is for non-negative integer/count features; f_classif is for continuous numerical features. FIFA attributes are continuous so f_classif is the correct choice here.

5. **RFECV confirming SelectKBest** — when RFECV keeps all features that SelectKBest chose, it validates that the statistical test and the model-based test agree. Strong signal that the feature set is genuinely optimal.
