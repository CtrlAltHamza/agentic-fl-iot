"""Data pipeline: synthetic HAR-like sensor data (default) or UCI-HAR if present."""
import os
import numpy as np


def make_synthetic_har(n_samples=6000, n_features=48, n_classes=6, seed=0):
    """Gaussian class clusters that mimic windowed accelerometer/gyro features."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 0.45, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    X = centers[y] + rng.normal(0, 2.2, size=(n_samples, n_features))
    return X.astype(np.float32), y.astype(np.int64)


def load_uci_har_split(root="data/raw/UCI HAR Dataset"):
    """Load UCI-HAR while preserving its official train/test split."""
    paths = {
        "X_train": os.path.join(root, "train", "X_train.txt"),
        "y_train": os.path.join(root, "train", "y_train.txt"),
        "X_test": os.path.join(root, "test", "X_test.txt"),
        "y_test": os.path.join(root, "test", "y_test.txt"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    Xtr = np.loadtxt(paths["X_train"]).astype(np.float32)
    ytr = np.loadtxt(paths["y_train"]).astype(np.int64) - 1
    Xte = np.loadtxt(paths["X_test"]).astype(np.float32)
    yte = np.loadtxt(paths["y_test"]).astype(np.int64) - 1
    return Xtr, ytr, Xte, yte


def load_uci_har(root="data/raw/UCI HAR Dataset"):
    """Compatibility loader returning concatenated official train and test data."""
    split = load_uci_har_split(root)
    if split is None:
        return None
    Xtr, ytr, Xte, yte = split
    return np.vstack([Xtr, Xte]), np.concatenate([ytr, yte])


def fit_standardizer(X_train):
    """Fit normalization on training data only to avoid test-set leakage."""
    mean = X_train.mean(0)
    std = X_train.std(0) + 1e-6
    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def standardize(X):
    """Compatibility helper; prefer fit_standardizer/apply_standardizer in experiments."""
    mean, std = fit_standardizer(X)
    return apply_standardizer(X, mean, std)


def train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


def dirichlet_partition(y, n_clients, alpha=0.3, seed=0):
    """Non-IID split: smaller alpha = more label skew across clients."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        props = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx, cuts)):
            parts[i].extend(chunk.tolist())
    return [np.array(p, dtype=int) for p in parts]
