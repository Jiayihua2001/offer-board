import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    ls = log_softmax(logits / T)
    return float(-ls[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0, tol: float = 1e-4) -> float:
    def f(log_t):
        return nll(logits, y, float(np.exp(log_t)))

    # Coarse grid in log T to find the valley, then golden-section search inside the bracket around it.
    grid = np.linspace(np.log(lo), np.log(hi), 50)
    vals = np.array([f(g) for g in grid])
    i = int(vals.argmin())
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    while (b - a) > tol / max(np.exp(b), 1.0) * 0.5:            # bracket width in T is about exp(b) * (b - a)
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2.0))


def calibration_bins(probs: np.ndarray, y: np.ndarray, n_bins: int):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    # Bin b covers (b / n_bins, (b + 1) / n_bins]; a confidence of exactly 1 lands in the last bin.
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    safe = np.maximum(count, 1)
    return acc_sum / safe, conf_sum / safe, count


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    acc, conf, count = calibration_bins(probs, y, n_bins)
    return float(np.sum(count / len(y) * np.abs(acc - conf)))
