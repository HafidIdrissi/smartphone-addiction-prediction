# Notebook Guide

This guide catalogs every Jupyter notebook in the repository to help contributors select the appropriate entry point, understand required resources, and navigate original versus mixed-source experiments without running expensive computations first.

---

## 🧭 Recommended Contributor Reading Order

For a clear progression through the methodology and experimental findings, we recommend exploring the notebooks in the following order:

1. [`notebooks/01_baseline.ipynb`](../notebooks/01_baseline.ipynb): **Foundation** — Start here to understand the dataset, initial behavioral ratio features, and the shared stratified five-fold cross-validation setup.
2. [`notebooks/02_exact_value_target_encoding.ipynb`](../notebooks/02_exact_value_target_encoding.ipynb): **Core Methodology (V3)** — Inspect the leak-free, fold-nested target encoding that handles repeated exact values without label leakage.
3. [`kaggle/predicting_smartphone_addiction_validation_first.ipynb`](../kaggle/predicting_smartphone_addiction_validation_first.ipynb): **Kaggle Deployment** — Review the self-contained dual-runtime implementation prepared for Kaggle environments.
4. [`notebooks/03_nested_oof_stack_v5.ipynb`](../notebooks/03_nested_oof_stack_v5.ipynb): **Ensemble Stacking (V5)** — Explore out-of-fold blending, model-correlation auditing, and stability-aware greedy selection.
5. [`experiments/v6_raw_highbin_xgboost.ipynb`](../experiments/v6_raw_highbin_xgboost.ipynb): **Controlled Diversity** — Review negative controls comparing higher histogram resolution (`max_bin=1024` vs `256`).
6. [`experiments/v7_oof_diversity.ipynb`](../experiments/v7_oof_diversity.ipynb): **Diversity Audit & Gating** — Inspect RealMLP and CatBoost lattice evaluations against the precommitted acceptance threshold (`+0.00005` OOF).

---

## 📊 Complete Notebook Catalog

| Notebook | Purpose | Expected Input | Important Outputs | Hardware | Runtime | Provenance |
|---|---|---|---|---|---|---|
| [`notebooks/01_baseline.ipynb`](../notebooks/01_baseline.ipynb) | CatBoost, LightGBM, and XGBoost baseline with behavioral ratios and frozen 5-fold CV | Raw competition data (`train.csv`, `test.csv`) | Aligned 5-fold OOF predictions, baseline fold metrics | CPU / GPU (T4 optional) | `medium` | `original` |
| [`notebooks/02_exact_value_target_encoding.ipynb`](../notebooks/02_exact_value_target_encoding.ipynb) | Leak-free nested target encoding for repeated exact values and GPU-accelerated XGBoost | Raw competition data (`train.csv`, `test.csv`) + public source dataset | Fold metrics, model scores, final OOF predictions, and run summary | GPU (T4 recommended) | `medium` | `original` |
| [`notebooks/03_nested_oof_stack_v5.ipynb`](../notebooks/03_nested_oof_stack_v5.ipynb) | Stability-aware greedy selection and rank-based stacking over original and audited public OOFs | Competition data (`train.csv`, `test.csv`) + public source dataset (trains original ensemble members) with optional imported public OOF members | V5 ensemble weights, blended OOF predictions, final submission file, model scores | GPU (T4 required for Lookup Transformer) | `long` (full run; cached reruns faster) | `mixed-source` |
| [`experiments/v6_raw_highbin_xgboost.ipynb`](../experiments/v6_raw_highbin_xgboost.ipynb) | Controlled diversity experiment studying histogram resolution (`max_bin=1024` vs `256`) as an encoding-starved control | Raw competition data (`train.csv`, `test.csv`) | Paired-fold OOF scores and comparison against credited community anchor (`submission_s6e8_community_blend.csv`) | GPU (T4 recommended) | `medium` | `original` |
| [`experiments/v7_oof_diversity.ipynb`](../experiments/v7_oof_diversity.ipynb) | Tabular neural network (RealMLP) and CatBoost lattice evaluation against strict gating criteria | Raw competition data (`train.csv`, `test.csv`), imported V5/V6 artifacts, and optional credited community anchor | RealMLP/CatBoost OOFs, rank-correlation matrix, gate verdict | GPU (T4 recommended) | `medium` | `mixed-source`* |
| [`kaggle/predicting_smartphone_addiction_validation_first.ipynb`](../kaggle/predicting_smartphone_addiction_validation_first.ipynb) | Self-contained, dual-runtime Kaggle edition replicating the V3 pipeline | Kaggle competition data (`/kaggle/input/playground-series-s6e8`) + public source dataset | In-notebook CV validation metrics and standalone submission | GPU (T4 / P100 on Kaggle) | `medium` | `original` |

\* *Note on `v7_oof_diversity.ipynb`*: Classified as `mixed-source` because it downloads competition data, imports prior V5 ensemble artifacts and V6 outputs, and can reference a credited community anchor for optional public-transfer screening (while the formal acceptance gate evaluates improvements against the V5 OOF reference), even though the newly trained RealMLP and CatBoost models themselves are original.

---

## 🏷️ Provenance Categories

- **`original`**: End-to-end model training and feature engineering executed entirely within this repository.
- **`mixed-source`**: Ensembles or comparative studies combining locally trained models with provenance-audited external out-of-fold prediction streams, or experiments importing upstream artifacts with credited community anchors.
- **`external/community`**: External prediction streams or benchmarks credited under open-source licenses and retained solely for reference without personal model claims.

---

## 💡 Notes on Execution

- **Kaggle Credentials**: Competitions data is retrieved at runtime using the Kaggle API. Ensure the `KAGGLE_API_TOKEN` secret is configured in Colab before execution.
- **No Data in Repository**: Raw competition datasets, checkpoints, and generated submission files are deliberately kept out of version control.
