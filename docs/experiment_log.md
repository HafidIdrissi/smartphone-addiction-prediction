# Experiment Log

This log records the main decisions rather than every temporary Colab run.

| Version | Main idea | Validation outcome | Public AUC | Decision |
|---|---|---|---:|---|
| Baseline | CatBoost, LightGBM, XGBoost, behavioral ratios | Established the shared five-fold workflow | Not promoted | Foundation |
| V3 | Exact-value frequency and fold-safe target encoding | Stronger repeat-value representation | Not promoted here | Retained |
| V5 | Stability-aware greedy selection over original and validated public OOF members | OOF AUC approximately `0.96952` | `0.97058` | Mixed-source experiment; provenance required |
| V6-256 | Raw, encoding-starved XGBoost control | Weaker than V5 | `0.96590` | Rejected |
| V6-1024 | Higher-resolution XGBoost histograms | Better than V6-256 on all 15 paired folds | `0.96694` | Scientifically positive, competitively insufficient |
| V7 RealMLP | Target-free neural tabular member | OOF AUC approximately `0.94067` | Not submitted | Rejected |
| V7 CatBoost lattice | Raw ordered values plus target-free exact categories | OOF AUC approximately `0.96716`; rank correlation to anchor about `0.98138` | Not submitted | Too weak for blend |
| V7 nested stack | V5 plus accepted diversity candidates | OOF gain approximately `+0.0000099`; 4/5 fold wins | Not submitted | Rejected by `0.00005` gate |
| Community blend | Fixed `0.45/0.55` percentile-rank blend of credited public predictions | No personal OOF claim | `0.97128` | Attribution-only reference |

## Decision rules

- Never compare models trained on unrelated folds as though their OOF predictions
  were aligned.
- Fit target encoders inside each training fold.
- Prefer fold-level stability to a tiny aggregate gain.
- Measure rank correlation before adding a model to an ROC AUC ensemble.
- Keep external prediction streams separate from personally trained artifacts.
- Do not promote a model based only on the public leaderboard.

## Post-competition lesson

The winning solution showed that a very strong individual RealMLP, built through
iterative error analysis and missingness-related feature engineering, contributed
more than repeatedly combining already-correlated public models. The next campaign
should therefore invest earlier in single-model ablations, missing-value structure,
and a disciplined experiment registry before expanding the ensemble.
