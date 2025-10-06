# Kaggle Competition – “Mechanisms of Action” (Top 2% Solution)
<img width="1296" height="297" alt="image" src="https://github.com/user-attachments/assets/0a1b11b6-f37f-40de-967d-2bf48cb419ce" />
https://www.kaggle.com/competitions/lish-moa

This repo contains my Top 2% (50th of 4,373 teams) solution to the Kaggle Mechanisms of Action (MoA) Prediction competition. I built a multilabel pipeline that ensembles Simple NNs (transfer-learning style, 4–5 hidden layers with label smoothing), TabNet, and a multi-input ResNet, trained with MultilabelStratifiedKFold, OneCycleLR, and extensive feature engineering (PCA features, VarianceThreshold selection, KMeans clustering on raw & PCA space, and stats features). Final blend used a weighted average (≈0.25 overall for the “final model” snapshot).
