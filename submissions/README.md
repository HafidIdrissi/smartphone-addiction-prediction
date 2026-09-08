# Submission Policy

Generated submission CSV files are intentionally excluded from this repository.

Reasons:

- competition data and generated predictions may have terms independent of this
  repository's MIT License;
- a CSV does not explain whether its signal was trained locally or imported;
- keeping predictions out of Git prevents accidental claims about community work.

## Recorded public results

| File | Public AUC | Provenance |
|---|---:|---|
| `submission_s6e8_v5_primary_nested.csv` | `0.97058` | Personally executed V5 stack containing original and credited public OOF members |
| `submission_s6e8_v6_e1_maxbin1024.csv` | `0.96694` | Personally trained V6 XGBoost |
| `submission_s6e8_v6_e1_maxbin256.csv` | `0.96590` | Personally trained V6 control |
| `submission_s6e8_community_blend.csv` | `0.97128` | Credited Apache-2.0 community prediction blend |

The notebooks validate identifier order, row count, column names, finite values,
and probability bounds before writing any submission.
