import numpy as np


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros(np.shape(a), dtype=float), where=b > 0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp                     # predicted as c but not c
    fn = cm.sum(axis=1) - tp                     # truly c but predicted otherwise
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    micro_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    micro_f1 = 0.0 if micro_p + micro_r == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)
    return {"precision": precision, "recall": recall, "f1": f1,
            "macro_f1": float(f1.mean()), "micro_f1": float(micro_f1)}


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last = counts.cumsum()                                   # 1-based rank of the last member of each tie group
    first = last - counts + 1
    ranks = ((first + last) / 2.0)[inv]                      # average rank within the tie group
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    # Mann-Whitney U: sum of positive ranks minus the ranks they would have if all positives came first.
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
