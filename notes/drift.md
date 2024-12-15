# Drift Notes

## Evidently AI

Todos os testes de drift: 
https://docs.evidentlyai.com/user-guide/customization/options-for-statistical-tests

For small data with <= 1000 observations in the reference dataset:

    For numerical columns (n_unique > 5): two-sample Kolmogorov-Smirnov test.

    For categorical columns or numerical columns with n_unique <= 5: chi-squared test.

    For binary categorical features (n_unique <= 2): proportion difference test for independent samples based on Z-score.

    All tests use a 0.95 confidence level by default.

For larger data with > 1000 observations in the reference dataset:

    For numerical columns (n_unique > 5):Wasserstein Distance.

    For categorical columns or numerical with n_unique <= 5):Jensen--Shannon divergence.

    All metrics use a threshold = 0.1 by default.