# Model results

Survey-weighted metrics on the 20% test split from `baseline_models.py`. The positive class is satisfied. Precision, recall, and F1 use the training-set Youden threshold. A cutoff of 0.50 predicts satisfied for every test row.

11/6 logistic regression on demographics was AUC 0.575. The 11/12 runs below use the same split and add the tree models.

| Model | Features | AUC | AUC 95% interval | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | Demographics | 0.575 | 0.519 to 0.627 | 0.893 | 0.603 | 0.720 |
| Logistic regression | Demographics + own earnings | 0.579 | 0.522 to 0.631 | 0.893 | 0.722 | 0.799 |
| Decision tree | Demographics | 0.563 | 0.512 to 0.618 | 0.887 | 0.552 | 0.681 |
| Decision tree | Demographics + own earnings | 0.567 | 0.509 to 0.630 | 0.896 | 0.665 | 0.763 |
| Random forest | Demographics | 0.567 | 0.512 to 0.622 | 0.891 | 0.567 | 0.693 |
| Random forest | Demographics + own earnings | 0.563 | 0.512 to 0.617 | 0.885 | 0.753 | 0.813 |
| Logistic regression | Demographics + earnings + hours, weeks, self-employment | 0.584 | 0.526 to 0.637 | 0.898 | 0.586 | 0.709 |
| Decision tree | Demographics + earnings + hours, weeks, self-employment | 0.615 | 0.557 to 0.670 | 0.907 | 0.646 | 0.754 |
| Random forest | Demographics + earnings + hours, weeks, self-employment | 0.600 | 0.551 to 0.648 | 0.896 | 0.734 | 0.807 |

## O*NET scores, 11/12

These rows come from `fit_models.py`. They use the same models and settings, but only the 8,385 rows with an O*NET occupation match. The earnings model is repeated on that sample, so compare within this table. Do not compare these AUC values with the full-sample table above.

| Model | Features | AUC | AUC 95% interval | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | Demographics + own earnings | 0.595 | 0.539 to 0.654 | 0.906 | 0.572 | 0.702 |
| Logistic regression | Demographics + own earnings + O*NET | 0.604 | 0.549 to 0.666 | 0.901 | 0.734 | 0.809 |
| Decision tree | Demographics + own earnings | 0.578 | 0.533 to 0.627 | 0.899 | 0.714 | 0.796 |
| Decision tree | Demographics + own earnings + O*NET | 0.593 | 0.544 to 0.645 | 0.905 | 0.700 | 0.789 |
| Random forest | Demographics + own earnings | 0.592 | 0.542 to 0.645 | 0.899 | 0.645 | 0.751 |
| Random forest | Demographics + own earnings + O*NET | 0.591 | 0.528 to 0.656 | 0.907 | 0.696 | 0.787 |
