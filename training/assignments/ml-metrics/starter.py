"""Classification metrics without sklearn. Vectorized NumPy only."""
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, C) integer counts; rows are true labels, columns predicted labels."""
    raise NotImplementedError


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """{"precision": (C,), "recall": (C,), "f1": (C,), "macro_f1": float, "micro_f1": float}; 0/0 counts as 0."""
    raise NotImplementedError


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2. Rank-based."""
    raise NotImplementedError


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy of P(y = 1) predictions p, clipped to [eps, 1 - eps]."""
    raise NotImplementedError
