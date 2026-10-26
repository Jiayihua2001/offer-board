# Classification metrics from scratch

Every applied-scientist loop has a round where you compute metrics by hand, and the follow-ups are where it gets interesting: which average to use for imbalanced classes, what AUC actually measures, why ties matter, and why log-loss needs clipping. Implement these in NumPy without loops over samples.

- `confusion_matrix(y_true, y_pred, n_classes)`: shape `(C, C)`, rows are true classes, columns predicted, integer counts. Labels are `0..C-1`.
- `precision_recall_f1(y_true, y_pred, n_classes)`: a dict with per-class arrays `"precision"`, `"recall"`, `"f1"` (shape `(C,)`), plus floats `"macro_f1"` (mean of per-class F1) and `"micro_f1"` (F1 computed from the pooled TP, FP, FN). Any `0 / 0` is defined as 0.
- `roc_auc(y_true, scores)`: binary labels in `{0, 1}` and real-valued scores. The probability that a uniformly random positive outscores a uniformly random negative, counting ties as one half. Rank-based, `O(n log n)`, no pairwise loop.
- `log_loss(y_true, p, eps=1e-15)`: mean binary cross-entropy of `P(y = 1)` predictions `p`, clipped to `[eps, 1 - eps]` first.

## Example

```
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])
confusion_matrix(y_true, y_pred, 3)   -> [[1, 1, 0],
                                          [0, 2, 0],
                                          [1, 0, 1]]
precision_recall_f1(...)["precision"] -> [0.5, 0.667, 1.0]
precision_recall_f1(...)["recall"]    -> [0.5, 1.0, 0.5]
precision_recall_f1(...)["micro_f1"]  -> 0.667   (= accuracy, 4 of 6)

roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))  -> 0.75
log_loss(np.array([1, 0]), np.array([0.9, 0.1]))                   -> 0.1054
```

## Constraints

- No loops over samples: the confusion matrix is one `np.add.at` or `bincount`; AUC uses ranks from `argsort`/`unique`.
- `roc_auc` must match an all-pairs reference exactly on data with many tied scores.
- `log_loss` must be finite for `p` exactly 0 or 1.

## What the interviewer listens for

- Why micro-F1 equals accuracy for single-label multi-class (every false positive for one class is a false negative for another), so macro-F1 is the one that reflects minority-class performance.
- The AUC-as-probability interpretation (Mann-Whitney U), which is why it is rank-based, insensitive to calibration, and why ties get half credit; and the trap of AUC on heavily imbalanced data, where precision-recall curves are more informative.
- Why log-loss must clip: a single confident wrong prediction is `-log(0) = inf`, and the clip value sets the maximum penalty a model can incur (`-log(1e-15) = 34.5`).
- Follow-up: how you would get confidence intervals for these metrics (bootstrap over samples) and when to threshold at something other than 0.5.
