"""Softmax regression in NumPy (light enough for constrained-device simulation)."""
import numpy as np


class SoftmaxModel:
    def __init__(self, n_features, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

    def get(self):
        return [self.W.copy(), self.b.copy()]

    def set(self, params):
        self.W, self.b = params[0].copy(), params[1].copy()

    def n_bytes(self):
        return self.W.nbytes + self.b.nbytes

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def loss_acc(self, X, y):
        p = self.predict_proba(X)
        loss = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
        return loss, float((p.argmax(1) == y).mean())

    def train_local(self, X, y, epochs=1, lr=0.1, batch=32, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                i = order[s:s + batch]
                p = self.predict_proba(X[i])
                p[np.arange(len(i)), y[i]] -= 1
                p /= len(i)
                self.W -= lr * (X[i].T @ p)
                self.b -= lr * p.sum(0)
