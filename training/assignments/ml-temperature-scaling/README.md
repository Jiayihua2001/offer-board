# Temperature scaling and expected calibration error

Calibration is the follow-up to any question about confidence thresholds, routing, or "when should the model abstain", and it is the method you shipped in the LLM router. Interviewers like it because it is one scalar, fitted by a one-dimensional search on held-out data, and it fixes a very visible failure (overconfident softmax outputs) without touching accuracy. Implement the fit and the metric that shows it worked.

The starter provides a stable `log_softmax` and `softmax`. Implement:

- `nll(logits, y, T) -> float`: mean negative log-likelihood of integer labels `y` under `softmax(logits / T)`. `logits` has shape `(N, C)`.
- `fit_temperature(logits, y, lo=0.05, hi=20.0, tol=1e-4) -> float`: the `T` in `[lo, hi]` that minimizes `nll`, found with a bracketing search (coarse grid plus golden-section or ternary refinement, or gradient descent on `log T`). Must be accurate to about `tol` in `T`.
- `calibration_bins(probs, y, n_bins) -> (bin_acc, bin_conf, bin_count)`: with `confidence = probs.max(1)` and `correct = probs.argmax(1) == y`, split `(0, 1]` into `n_bins` equal-width bins and return the mean accuracy, the mean confidence and the count per bin (arrays of shape `(n_bins,)`; empty bins have accuracy and confidence 0).
- `ece(probs, y, n_bins=15) -> float`: `sum over bins of (count / N) * |bin_acc - bin_conf|`.

## Example

```
probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])   # confidences .9 .8 .7 .6
y     = np.array([0, 1, 0, 0])                                        # correct: yes no yes yes
calibration_bins(probs, y, n_bins=4)   -> acc [0, 0, 1.0, 0.5], conf [0, 0, 0.65, 0.85], count [0, 0, 2, 2]
ece(probs, y, n_bins=4)                -> 0.5 * |1.0 - 0.65| + 0.5 * |0.5 - 0.85| = 0.35

logits = 3 * calibrated_logits         # overconfident by a factor of 3
fit_temperature(logits, y)             -> about 3.0
```

## Constraints

- No loops over samples; a loop over bins or over search iterations is fine.
- `nll` must be stable for logits of magnitude 1000 (use the provided `log_softmax`).
- Tests: `T` is recovered within 0.2 on synthetic data (sharpened and flattened logits), `argmax` is unchanged by scaling, and `ece` and `nll` both drop after fitting.

## What the interviewer listens for

- Why a single temperature cannot change accuracy: dividing logits by `T > 0` is monotone, so `argmax` is unchanged; it only moves probability mass toward or away from the top class.
- Why fit `T` on a held-out set with NLL (a proper scoring rule) and not on the training set, where the model is already overconfident, and why NLL and not ECE (ECE is piecewise constant in `T`, so it has no useful gradient).
- What ECE measures and its weaknesses: binning bias, sensitivity to `n_bins`, only the top-class confidence is checked. Know the reliability-diagram picture.
- Follow-up: how you extend this to a router threshold (calibrated probability of "small model is right" feeds a cost-aware decision rule), and when you need something more flexible than one scalar (vector or matrix scaling, isotonic regression, Platt scaling for binary scores).
