# Predicting Smartphone Addiction

An open, validation-first solution for the Kaggle Playground Series S6E8 binary
classification challenge. The project studies how behavioral features, repeated
exact values, missingness, diverse tabular models, and out-of-fold stacking affect
ROC AUC.

## Project highlights

- frozen stratified five-fold validation shared by every model;
- leak-free nested target encoding for repeated exact values;
- behavioral ratios and constrained screen-time features;
- CatBoost, LightGBM, XGBoost, RealMLP, and rank-based stacking;
- saved OOF predictions, fold diagnostics, and model-correlation audits;
- explicit separation between original models and community predictions;
- checkpointing designed for interrupted Google Colab sessions.

## Results

| Experiment | Evidence | Result | Interpretation |
|---|---|---:|---|
| V3 exact-value XGBoost | Original, repository-trained pipeline | OOF evaluated; public score not promoted | Clearest fully original portfolio model |
| V5 nested research stack | Original models plus provenance-audited public OOF members | Public AUC `0.97058` | Personally executed mixed-source experiment |
| Credited community rank blend | External Apache-2.0 prediction streams | Public AUC `0.97128` | Community experiment, not a personal model score |
| V6 XGBoost `max_bin=1024` | Personal Kaggle submission | Public AUC `0.96694` | More histogram resolution did not transfer well |
| V6 XGBoost `max_bin=256` | Personal Kaggle submission | Public AUC `0.96590` | Useful negative control |
| V7 diversity stack | Nested OOF comparison | `+0.0000099` OOF over V5 | Rejected by the precommitted acceptance threshold |

The public leaderboard used only a subset of the hidden labels. These values are
reported as experimental evidence, not as guarantees of generalization. V3 is the
clearest fully original portfolio pipeline. V5 and the `0.97128` community result
are intentionally labeled as mixed-source or external because some or all of their
predictive streams were not trained in this repository.

## Repository map

```text
notebooks/
  01_baseline.ipynb                    readable CatBoost/LightGBM/XGBoost baseline
  02_exact_value_target_encoding.ipynb leak-free exact-value GPU pipeline
  03_nested_oof_stack_v5.ipynb         provenance-audited mixed-source stack
experiments/
  v6_raw_highbin_xgboost.ipynb         controlled diversity experiment
  v7_oof_diversity.ipynb               RealMLP/CatBoost OOF audit
docs/
  notebook_guide.md                    guide to notebook roles, runtimes, and execution order
  technical_writeup.md                 detailed methodology and lessons
  experiment_log.md                    concise experiment history
  attribution.md                       source and license boundaries
kaggle/
  predicting_smartphone_addiction_validation_first.ipynb
                                        public, dual-runtime V3 edition
submissions/
  README.md                             provenance and publication policy
```

## Reproduce on Google Colab

1. Accept the competition rules on the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s6e8).
2. Create a Kaggle API token.
3. In Colab, create a secret named `KAGGLE_API_TOKEN` and enable notebook access.
4. Select a T4 GPU for the XGBoost and neural experiments.
5. Start with `notebooks/01_baseline.ipynb`, then run the later notebooks in order (refer to the [Notebook Guide](docs/notebook_guide.md) for roles, hardware expectations, and recommended execution flow).

The notebooks install their own competition dependencies and download the data at
runtime. Raw competition files, credentials, model checkpoints, and generated
submissions are deliberately excluded from Git.

For local exploration:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU availability and package compatibility vary across environments. Colab is the
reference runtime for the full experiments.

## Validation design

Every target-derived transformation is fitted inside the current training fold.
All candidate models use aligned OOF rows so that score deltas and correlations are
meaningful. Ensemble members are selected from OOF evidence rather than repeated
public-leaderboard probing.

This matters especially for exact-value target encoding. A global encoding would
allow validation labels to influence their own features and produce an optimistic
score. The nested procedure in this repository prevents that leakage.

## What did not work

The repository keeps two carefully measured unsuccessful directions:

- increasing XGBoost histogram resolution created a different model but remained
  too weak to improve the stronger anchor;
- adding RealMLP and a CatBoost lattice improved the V5 OOF score by only about
  `0.00001`, below the predefined minimum improvement of `0.00005`.

Publishing negative results makes the research auditable and prevents the same
expensive experiments from being repeated without evidence.

## Responsible use

The dataset is synthetic and the target is a competition label. Predictions from
this project are not a medical diagnosis and should not be used to make decisions
about an individual. Real-world use would require consent, domain validation,
fairness analysis, calibration, privacy protections, and independent data.

## Attribution

The competition was created by Kaggle as Playground Series Season 6 Episode 8.
Public research by Kodai Fukuda, Tamerlan Omralinov, Dariush Afshar, Szymon
Klapinski, and Souvik D. Biswas informed parts of the experimental direction.

The community score used Apache-2.0 prediction streams published by Souvik D.
Biswas. Those files are not redistributed here and the result must not be described
as a score produced entirely by the author's models.

## License

The original code and documentation in this repository are released under the MIT
License. Kaggle competition data and third-party artifacts are not covered by this
license and remain subject to their respective terms.
