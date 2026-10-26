import numpy as np

from solution import confusion_matrix, log_loss, precision_recall_f1, roc_auc

Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])


def auc_pairs(y, s):
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def test_confusion_matrix_hand_example():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.shape == (3, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert cm.sum() == 6
    assert np.array_equal(cm.sum(axis=1), np.bincount(Y_TRUE, minlength=3)), "row sums are the true class counts"
    empty = confusion_matrix(np.array([0, 0]), np.array([0, 0]), 3)
    assert empty.tolist() == [[2, 0, 0], [0, 0, 0], [0, 0, 0]], "classes that never appear still get rows"


def test_precision_recall_f1_hand_example():
    m = precision_recall_f1(Y_TRUE, Y_PRED, 3)
    assert np.allclose(m["precision"], [0.5, 2 / 3, 1.0])
    assert np.allclose(m["recall"], [0.5, 1.0, 0.5])
    assert np.allclose(m["f1"], [0.5, 0.8, 2 / 3])
    assert np.isclose(m["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
    assert np.isclose(m["micro_f1"], 4 / 6)


def test_zero_division_gives_zero():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0])             # classes 1 and 2 never predicted
    m = precision_recall_f1(y_true, y_pred, 3)
    assert np.all(np.isfinite(m["precision"])) and np.all(np.isfinite(m["f1"]))
    assert np.allclose(m["precision"], [0.5, 0.0, 0.0])
    assert np.allclose(m["recall"], [1.0, 0.0, 0.0])
    assert np.allclose(m["f1"], [2 / 3, 0.0, 0.0])
    assert np.isclose(m["macro_f1"], 2 / 9)


def test_micro_f1_is_accuracy_and_macro_is_mean_of_per_class():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 4, size=500))
    m = precision_recall_f1(y_true, y_pred, 4)
    assert np.isclose(m["micro_f1"], np.mean(y_true == y_pred))
    assert np.isclose(m["macro_f1"], m["f1"].mean())
    cm = confusion_matrix(y_true, y_pred, 4)
    tp = np.diag(cm)
    assert np.allclose(m["precision"], tp / cm.sum(axis=0))
    assert np.allclose(m["recall"], tp / cm.sum(axis=1))


def test_roc_auc_hand_values():
    assert np.isclose(roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])), 0.75)
    assert np.isclose(roc_auc(np.array([0, 1, 0, 1]), np.array([0.2, 0.9, 0.3, 0.8])), 1.0)
    assert np.isclose(roc_auc(np.array([0, 1, 0, 1]), np.array([0.9, 0.2, 0.8, 0.3])), 0.0)
    assert np.isclose(roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])), 0.5), "all tied -> 0.5"
    assert np.isclose(roc_auc(np.array([1, 0, 0]), np.array([0.5, 0.5, 0.1])), 0.75), "one tie counts half"


def test_roc_auc_matches_pairwise_reference_with_ties():
    rng = np.random.default_rng(1)
    for trial in range(5):
        y = (rng.random(80) < 0.35).astype(int)
        s = np.round(rng.random(80) + 0.4 * y, 1)          # one decimal -> many ties
        assert np.isclose(roc_auc(y, s), auc_pairs(y, s), atol=1e-12), f"trial {trial}"
    s = rng.normal(size=80)
    assert np.isclose(roc_auc(y, s), auc_pairs(y, s), atol=1e-12)


def test_log_loss_values_and_clipping():
    assert np.isclose(log_loss(np.array([1, 0]), np.array([0.9, 0.1])), -np.log(0.9))
    assert np.isclose(log_loss(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5])), np.log(2.0))
    extreme = log_loss(np.array([0, 1]), np.array([1.0, 0.0]))
    assert np.isfinite(extreme) and 30.0 < extreme < 40.0, "confident and wrong is capped by the clip, not infinite"
    assert np.isclose(log_loss(np.array([0, 1]), np.array([1.0, 0.0]), eps=1e-6), -np.log(1e-6), rtol=1e-6)
    assert log_loss(np.array([1, 1]), np.array([1.0, 1.0])) < 1e-12
