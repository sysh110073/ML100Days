"""兩套框架共用的離線二元分類資料；只用 NumPy，避免框架相互依賴。"""

import numpy as np


def make_binary_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(800, 2)).astype("float32")
    y = (x[:, 0] + 0.7 * x[:, 1] > 0).astype("int32")
    order = rng.permutation(len(x))
    train_idx, valid_idx = order[:640], order[640:]
    return x[train_idx], y[train_idx], x[valid_idx], y[valid_idx]
