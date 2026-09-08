# Predicting Smartphone Addiction: Feature Engineering and Model Blending

## TL;DR

This project explores a binary classification problem from Kaggle Playground Series S6E8: predicting whether a person is at risk of smartphone addiction from behavioral and demographic variables.

My main objective was not to chase small public-leaderboard fluctuations. I wanted to build a reproducible pipeline that could answer three questions:

1. How can repeated exact values in synthetic tabular data be used without leaking the target?
2. Which behavioral feature relationships are both useful and interpretable?
3. Can models with genuinely different representations improve an out-of-fold ensemble?

The resulting pipeline combines fold-safe exact-value target encoding, frequency and behavioral features, external reference-distribution statistics, gradient-boosted trees, a compact neural tabular model, and nested out-of-fold stacking.

---

## 1. Competition and Evaluation

The target is `addicted_label`, and each test row requires a probability rather than a hard class prediction. Submissions are evaluated with ROC AUC.

ROC AUC measures how well the model ranks positive examples above negative examples. This influenced two important choices:

- every model was evaluated with out-of-fold ROC AUC;
- rank-based predictions were considered during blending because probability scales can differ across model families.

The identifier column was excluded from training because it has no defensible behavioral meaning.

Competition: [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)

---

## 2. Data

The dataset contains twelve input variables.

### Numerical variables

- age;
- daily screen time;
- social-media usage;
- gaming time;
- work or study time;
- sleep duration;
- notifications per day;
- application opens per day;
- weekend screen time.

### Categorical variables

- gender;
- stress level;
- academic or work impact.

The competition data is synthetic but was inspired by a public smartphone-usage dataset. Synthetic tabular data can preserve real-world relationships while also introducing repeated values and generator-specific structure. Those properties became part of the modeling strategy.

The public source dataset was used only as a reference distribution. Exact overlaps with competition training rows were removed, and the source rows were not blindly appended to the training data.

---

## 3. Validation Before Modeling

I froze one stratified five-fold cross-validation split and reused it for every model family.

This is important for honest model comparison and stacking. If different models use unrelated validation folds, their out-of-fold predictions are not aligned and the apparent ensemble performance can become misleading.

The validation rules were:

- preserve the target ratio in every fold;
- fit every target-derived transformation inside the current training fold;
- generate exactly one out-of-fold prediction for every training row;
- select models and blend weights from OOF evidence, not repeated public submissions;
- save folds, configurations, predictions, and scores for reproducibility.

This design also protects the experiment from a common mistake: global target encoding. Encoding a value with labels from the full training set would allow validation labels to influence their own features and inflate the local score.

---

## 4. Interpretable Behavioral Features

The first feature group describes how a person's available time is distributed. Instead of generating hundreds of arbitrary polynomial interactions, I created a compact set of ratios and differences with a behavioral interpretation.

Examples include:

- social-media time as a share of total screen time;
- gaming time as a share of total screen time;
- work or study time as a share of total screen time;
- screen-time-to-sleep ratio;
- social-media-to-sleep ratio;
- screen-time-to-work ratio;
- weekend-screen-time-to-sleep ratio;
- screen time minus work or study time;
- screen time as a share of waking hours;
- the number of missing values in a row.

I also decomposed daily screen time into social media, gaming, work or study, and unallocated time. These features help a model distinguish two people with similar total screen time but very different usage patterns.

---

## 5. Learning From Repeated Exact Values

Several numerical columns contain many repeated exact values. Treating them only as continuous quantities can discard information about the synthetic generation process.

### Frequency encoding

For each original variable, I measured how frequently its exact value appeared in the combined train and test feature distribution.

This transformation is unsupervised: it uses no labels and therefore does not leak the target. It tells the model whether a value is common, rare, or potentially characteristic of a particular generation regime.

### Fold-safe exact-value target encoding

Target encoding estimates the positive-class rate associated with an exact feature value. It can be powerful here, but it requires stricter validation.

Inside every outer fold, I used another stratified cross-fitting procedure:

1. split the outer training partition into inner folds;
2. build encodings for each inner validation partition using only the other inner partitions;
3. fit the final mapping on the complete outer training partition;
4. transform the outer validation and test rows with that mapping;
5. shrink rare groups toward the global target mean.

The outer validation labels are never used to construct their own encodings. This nested procedure is slower than global encoding, but its OOF score is meaningful.

### Pair keys

I kept only a small number of interpretable exact-value pairs. Limiting pair keys controls sparsity and reduces the chance of learning accidental combinations.

---

## 6. External Reference-Distribution Features

The original public dataset provides useful information about the source distribution. After removing overlaps, I extracted compact statistics such as:

- empirical CDF position;
- differences between class-conditional CDFs;
- distance to global and class-specific medians;
- coarse, smoothed target-rate curves.

These features describe where a competition value lies relative to the source population. The reference transformations never use competition validation labels.

This distinction matters: external data can improve a model, but only when its origin, license, overlap checks, and role in validation are documented.

---

## 7. Complementary Model Families

### XGBoost with exact-value encodings

XGBoost was the main tree-based learner. Two variants used the same core features but different seeds and tree-growth configurations. Reusing fold-safe target encodings reduced preprocessing cost while the model variants contributed some ranking diversity.

GPU acceleration made the full five-fold, multi-seed experiment practical on a Google Colab T4.

### CatBoost with native categorical variables

CatBoost received the three genuine categorical variables as categories and preserved the numerical variables as ordered quantities.

I deliberately avoided converting every repeated floating-point value into an unordered category. That would retain identity but destroy useful numerical order. CatBoost therefore offered a representation different from the exact-encoding XGBoost models.

### Class-conditional histogram model

A lightweight histogram model estimated smoothed class-conditional likelihoods for individual features. Its standalone AUC was not expected to beat the strongest boosted tree. Its purpose was to contribute a different error pattern to the ensemble.

### Compact Lookup Transformer

The neural member represented each feature with:

- an exact-value embedding;
- a learned column embedding;
- a normalized numerical value;
- a missingness indicator;
- periodic sine and cosine numerical features.

Rare and unseen identities were mapped to an unknown token. A small Transformer learned interactions between the twelve feature tokens. This architecture preserved both exact identity and numerical order while remaining small enough for a Colab T4.

---

## 8. OOF Auditing and Model Blending

A good ensemble needs strong models, but strength alone is not enough. If every model produces nearly the same ranking, averaging them adds little value.

For every member, I recorded:

- overall OOF ROC AUC;
- fold-level AUC;
- Spearman correlation with other predictions;
- runtime and configuration;
- complete OOF and test predictions.

I converted predictions to percentile ranks before stacking. This places model families on a comparable scale and matches the ranking nature of ROC AUC. Clipped prediction logits were also included to retain confidence information.

The meta-model was trained out of fold. A plain stack was combined with a small regime-aware stack that considered missingness and model disagreement. The final mixture was specified in advance instead of searching many weights on the public leaderboard.

The submission pipeline then verified:

- the official identifier order;
- the expected row count and column names;
- finite predictions with no missing values;
- probabilities inside the valid range.

---

## 9. Results and Honest Attribution

The main technical result of this project is the validation framework and independently trained model library. It produces aligned OOF predictions, fold-level diagnostics, model-correlation reports, and validated submission candidates.

In a separate leaderboard experiment, a **credited community rank blend** reached a public ROC AUC of **0.97128**. That file combined public Apache-2.0 prediction streams and must not be described as a score produced entirely by my personal models.

I keep these two results distinct:

- the original modeling pipeline demonstrates my feature engineering, validation, training, and stacking work;
- the community blend demonstrates responsible reuse, rank blending, provenance tracking, and attribution.

This separation is more valuable than presenting a leaderboard number without explaining where its predictive signal came from.

---

## 10. What I Learned

### Validation quality is part of the model

The strongest feature is not useful if its validation procedure leaks information. Nested target encoding and aligned folds made the experiment slower, but also made the conclusions defensible.

### Synthetic data rewards structural analysis

Repeated exact values, constrained time budgets, and relationships with the source distribution carried more signal than a large collection of arbitrary interactions.

### Diversity must be measured

A weaker but decorrelated model can help an ensemble more than another near-copy of the best tree model. OOF AUC and pairwise rank correlation should be examined together.

### Public leaderboard gains can be deceptive

Small public-score differences may reflect noise or overfitting to the visible test subset. Fold stability, provenance, and private-leaderboard diversity are safer criteria for selecting final submissions.

### Reproducibility creates portfolio value

Saving folds, OOF predictions, test predictions, configurations, and checksums makes the work explainable in an interview and reusable after a Colab disconnection.

---

## 11. Limitations and Next Steps

The competition dataset is synthetic, so performance does not automatically imply clinical or behavioral validity in the real world. Smartphone addiction is also a sensitive concept that cannot be reduced to a single model prediction without domain expertise and careful data collection.

The next experiments I would prioritize are:

- repeated cross-validation to estimate score variance;
- constrained optimization of ensemble weights using only OOF predictions;
- calibration analysis in addition to ranking performance;
- subgroup error analysis across age and gender;
- ablation studies for exact-value, reference-distribution, and behavioral features;
- validation on an independent real-world dataset.

---

## Conclusion

This project began as a Kaggle classification challenge, but the most useful outcome was a complete experimental workflow: inspect the data-generating structure, engineer interpretable signals, prevent leakage, train diverse models on aligned folds, and blend them using out-of-fold evidence.

The leaderboard is one measurement. The reproducible reasoning behind the score is the part I want this project to demonstrate.

---

## Acknowledgements

Public S6E8 research by Kodai Fukuda, Tamerlan Omralinov, Dariush Afshar, Szymon Klapinski, and Souvik D. Biswas informed parts of the experimental direction. Public prediction files are kept separate from original model outputs, and their provenance and licenses are recorded in the notebooks.

