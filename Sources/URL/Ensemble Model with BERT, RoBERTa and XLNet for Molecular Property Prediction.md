---
title: "Ensemble Model with BERT, RoBERTa and XLNet for Molecular Property Prediction"
source: "https://www.icck.org/article/abs/tetai.2026.604672"
author:
  - "[[Junling Hu]]"
published: 2026-08-06
created: 2026-09-30
description: "Molecular property prediction is a fundamental task in drug discovery and materials science, yet most high-performing approaches depend on large-scale pretraining that demands substantial computational resources. This work proposes a pretraining-free ensemble framework that trains multiple Transformer-based architectures—BERT, RoBERTa, and XLNet—from random initialization using the Atom-in-SMILES (AIS) molecular representation, which provides richer atomic-level semantics than conventional SMILES. The three Transformer encoders are coupled with BiLSTM prediction heads and integrated via a BaggingRegressor to reduce variance and improve generalization. Experiments on the ZINC250k and ZINC310k benchmarks demonstrate that the proposed framework achieves competitive performance against pretrained baselines including GROVER, CHEM-BERT, and D-MPNN, while requiring only task-specific end-to-end training with adaptive early stopping. These results establish that carefully designed molecular representations combined with heterogeneous ensemble learning can serve as a practical and resource-efficient alternative to pretraining-based paradigms in molecular modeling."
tags:
  - "clippings"
---
APA Style

Hu, J. (2026). Ensemble Model with BERT, RoBERTa and XLNet for Molecular Property Prediction. ICCK Transactions on Emerging Topics in Artificial Intelligence, 3(3), 170-187. https://doi.org/10.62762/TETAI.2026.604672

Export Citation

RIS Format

Compatible with EndNote, Zotero, Mendeley, and other reference managers

```
TY  - JOUR
AU  - Hu, Junling
PY  - 2026
DA  - 2026/08/06
TI  - Ensemble Model with BERT, RoBERTa and XLNet for Molecular Property Prediction
JO  - ICCK Transactions on Emerging Topics in Artificial Intelligence
T2  - ICCK Transactions on Emerging Topics in Artificial Intelligence
JF  - ICCK Transactions on Emerging Topics in Artificial Intelligence
VL  - 3
IS  - 3
SP  - 170
EP  - 187
DO  - 10.62762/TETAI.2026.604672
UR  - https://www.icck.org/article/abs/TETAI.2026.604672
KW  - ensemble learning
KW  - BERT
KW  - RoBERTa
KW  - XLNet
KW  - molecular property prediction
AB  - Molecular property prediction is a fundamental task in drug discovery and materials science, yet most high-performing approaches depend on large-scale pretraining that demands substantial computational resources. This work proposes a pretraining-free ensemble framework that trains multiple Transformer-based architectures—BERT, RoBERTa, and XLNet—from random initialization using the Atom-in-SMILES (AIS) molecular representation, which provides richer atomic-level semantics than conventional SMILES. The three Transformer encoders are coupled with BiLSTM prediction heads and integrated via a BaggingRegressor to reduce variance and improve generalization. Experiments on the ZINC250k and ZINC310k benchmarks demonstrate that the proposed framework achieves competitive performance against pretrained baselines including GROVER, CHEM-BERT, and D-MPNN, while requiring only task-specific end-to-end training with adaptive early stopping. These results establish that carefully designed molecular representations combined with heterogeneous ensemble learning can serve as a practical and resource-efficient alternative to pretraining-based paradigms in molecular modeling.
SN  - 3068-6652
PB  - Institute of Central Computation and Knowledge
LA  - English
ER  -
```

BibTeX Format

Compatible with LaTeX, BibTeX, and other reference managers

```
@article{Hu2026Ensemble,
  author = {Junling Hu},
  title = {Ensemble Model with BERT, RoBERTa and XLNet for Molecular Property Prediction},
  journal = {ICCK Transactions on Emerging Topics in Artificial Intelligence},
  year = {2026},
  volume = {3},
  number = {3},
  pages = {170-187},
  doi = {10.62762/TETAI.2026.604672},
  url = {https://www.icck.org/article/abs/TETAI.2026.604672},
  abstract = {Molecular property prediction is a fundamental task in drug discovery and materials science, yet most high-performing approaches depend on large-scale pretraining that demands substantial computational resources. This work proposes a pretraining-free ensemble framework that trains multiple Transformer-based architectures—BERT, RoBERTa, and XLNet—from random initialization using the Atom-in-SMILES (AIS) molecular representation, which provides richer atomic-level semantics than conventional SMILES. The three Transformer encoders are coupled with BiLSTM prediction heads and integrated via a BaggingRegressor to reduce variance and improve generalization. Experiments on the ZINC250k and ZINC310k benchmarks demonstrate that the proposed framework achieves competitive performance against pretrained baselines including GROVER, CHEM-BERT, and D-MPNN, while requiring only task-specific end-to-end training with adaptive early stopping. These results establish that carefully designed molecular representations combined with heterogeneous ensemble learning can serve as a practical and resource-efficient alternative to pretraining-based paradigms in molecular modeling.},
  keywords = {ensemble learning, BERT, RoBERTa, XLNet, molecular property prediction},
  issn = {3068-6652},
  publisher = {Institute of Central Computation and Knowledge}
}
```