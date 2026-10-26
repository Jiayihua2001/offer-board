import numpy as np


def init_params(d_in: int, d_hidden: int, d_out: int, rng: np.random.Generator) -> dict:
    return {
        "W1": rng.normal(size=(d_in, d_hidden)) * np.sqrt(2.0 / d_in),
        "b1": np.zeros(d_hidden),
        "W2": rng.normal(size=(d_hidden, d_out)) * np.sqrt(2.0 / d_hidden),
        "b2": np.zeros(d_out),
    }


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def forward(X: np.ndarray, params: dict):
    h_pre = X @ params["W1"] + params["b1"]
    h = np.maximum(h_pre, 0.0)
    logits = h @ params["W2"] + params["b2"]
    return logits, (X, h_pre, h)


def loss_and_grads(X: np.ndarray, y: np.ndarray, params: dict):
    logits, (X, h_pre, h) = forward(X, params)
    n = X.shape[0]
    ls = log_softmax(logits)
    loss = float(-ls[np.arange(n), y].mean())

    dlogits = np.exp(ls)                         # softmax probabilities
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n                                 # d(mean loss)/d(logits) = (p - onehot) / N
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params["W2"].T
    dh_pre = dh * (h_pre > 0)                    # ReLU passes gradient only where it was active
    dW1 = X.T @ dh_pre
    db1 = dh_pre.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(X: np.ndarray, y: np.ndarray, params: dict, lr: float = 0.5, n_iter: int = 300):
    params = {k: v.copy() for k, v in params.items()}
    losses = np.empty(n_iter)
    for t in range(n_iter):
        loss, grads = loss_and_grads(X, y, params)
        losses[t] = loss
        for k in params:
            params[k] = params[k] - lr * grads[k]
    return params, losses


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    logits, _ = forward(X, params)
    return logits.argmax(axis=1)
