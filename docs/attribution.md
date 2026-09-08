# Attribution and Provenance

## Competition

- Kaggle Playground Series S6E8: [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)
- Metric: ROC AUC
- Competition data is not redistributed by this repository.

## Original work in this repository

The baseline, exact-value encoding pipeline, behavioral features, training wrappers,
validation audits, experiment gates, and locally generated model predictions were
assembled for this project. V3 is the cleanest notebook to cite when discussing a
fully repository-trained model.

## Public research and prediction sources

Public work by the following Kaggle contributors informed feature engineering,
validation, and ensemble experiments:

- Kodai Fukuda;
- Tamerlan Omralinov;
- Dariush Afshar;
- Szymon Klapinski;
- Souvik D. Biswas.

The credited community rank blend was based on Apache-2.0 prediction streams from
Souvik D. Biswas:

- [S6E8 Top 20 Formula: Dual Master Rank Blend](https://www.kaggle.com/code/souvikdbiswas/s6e8-top-20-formula-dual-master-rank-blend)

V5 optionally imports allowlisted public OOF libraries. It validates row alignment,
excludes disallowed or level-2 members, and exports a provenance table. Therefore,
V5 is a mixed-source experiment and must not be described as fully self-trained.

## Winning-solution retrospective

The post-competition interpretation refers to Chris Deotte's published writeup:

- [1st Place - Distributed Intelligence - NVIDIA Inference Hub](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe)

That writeup is cited for comparison and learning. Its models, code, and unpublished
feature engineering are not part of this repository.

## License boundary

The MIT License applies only to original code and documentation contained here. It
does not relicense Kaggle data, public prediction files, third-party datasets,
notebooks, model weights, or other contributors' work.

