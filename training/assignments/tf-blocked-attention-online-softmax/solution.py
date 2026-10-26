import numpy as np


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    s = q @ k.T / np.sqrt(q.shape[-1])
    m = s.max(axis=1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(axis=1, keepdims=True)
    return (p / l) @ v, (m + np.log(l))[:, 0]


def blocked_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_size: int):
    T_q, d = q.shape
    T_k = k.shape[0]
    scale = 1.0 / np.sqrt(d)
    m = np.full(T_q, -np.inf)           # running row max
    l = np.zeros(T_q)                   # running softmax denominator
    acc = np.zeros((T_q, v.shape[1]))   # running unnormalised output
    for start in range(0, T_k, block_size):
        k_blk, v_blk = k[start:start + block_size], v[start:start + block_size]
        s = q @ k_blk.T * scale                      # (T_q, b)
        m_new = np.maximum(m, s.max(axis=1))
        corr = np.exp(m - m_new)                     # rescale what was accumulated under the old max
        p = np.exp(s - m_new[:, None])
        l = l * corr + p.sum(axis=1)
        acc = acc * corr[:, None] + p @ v_blk
        m = m_new
    return acc / l[:, None], m + np.log(l)
