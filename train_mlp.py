import numpy as np
from scipy import optimize


# ── Activation ───────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_gradient(z):
    g = sigmoid(z)
    return g * (1 - g)


# ── Cost function with backpropagation ───────────────────────────────────────

def nn_cost_function(nn_params, layer_sizes, X, y, lambda_):
    """
    Compute the regularised cross-entropy cost and its gradient.

    Parameters
    ----------
    nn_params    : flat parameter vector (all weight matrices concatenated)
    layer_sizes  : list of layer widths [input, hidden..., output]
    X            : (m, n) feature matrix
    y            : (m,) integer label vector
    lambda_      : L2 regularisation coefficient

    Returns
    -------
    J    : scalar cost
    grad : flat gradient vector (same shape as nn_params)
    """
    m          = X.shape[0]
    num_labels = layer_sizes[-1]

    # Reconstruct weight matrices from flat vector
    thetas, idx = [], 0
    for i in range(len(layer_sizes) - 1):
        size = layer_sizes[i + 1] * (layer_sizes[i] + 1)
        thetas.append(nn_params[idx:idx + size].reshape(layer_sizes[i + 1], layer_sizes[i] + 1))
        idx += size

    # Forward pass
    a  = [np.concatenate([np.ones((m, 1)), X], axis=1)]
    zs = []
    for i, theta in enumerate(thetas):
        z      = a[i] @ theta.T
        zs.append(z)
        a_next = sigmoid(z)
        if i < len(thetas) - 1:
            a_next = np.concatenate([np.ones((m, 1)), a_next], axis=1)
        a.append(a_next)

    h        = a[-1]
    y_matrix = np.eye(num_labels)[y]

    # cost
    J = np.sum(-y_matrix * np.log(h + 1e-15) - (1 - y_matrix) * np.log(1 - h + 1e-15)) / m
    J += (lambda_ / (2 * m)) * sum(np.sum(t[:, 1:] ** 2) for t in thetas)

    # Backpropagation
    deltas = [h - y_matrix]
    for i in range(len(thetas) - 1, 0, -1):
        d = (deltas[0] @ thetas[i][:, 1:]) * sigmoid_gradient(zs[i - 1])
        deltas.insert(0, d)

    grads = []
    for i, theta in enumerate(thetas):
        g = deltas[i].T @ a[i] / m
        g[:, 1:] += (lambda_ / m) * theta[:, 1:]
        grads.append(g.ravel())

    return J, np.concatenate(grads)


# ── Model wrapper ────────────────────────────────────────────────────────────

class HandmadeMLP:

    def __init__(self, params, layer_sizes, classes):
        self.params      = params
        self.layer_sizes = layer_sizes
        self.classes_    = classes

    def predict_proba(self, X):
        """Forward pass — returns (n_samples, n_classes) probability matrix."""
        a, idx = X, 0
        thetas = []
        for i in range(len(self.layer_sizes) - 1):
            size = self.layer_sizes[i + 1] * (self.layer_sizes[i] + 1)
            thetas.append(self.params[idx:idx + size].reshape(
                self.layer_sizes[i + 1], self.layer_sizes[i] + 1))
            idx += size

        for t in thetas:
            a = sigmoid(np.concatenate([np.ones((a.shape[0], 1)), a], axis=1) @ t.T)
        return a

    def predict(self, X):
        """Return the class with the highest predicted probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── Training ─────────────────────────────────────────────────────────────────

def train_mlp(X_train, y_train,
              hidden_layers=(1024, 512, 256),
              max_iter=300,
              alpha=1e-4,
              random_state=42,
              verbose=True):

    np.random.seed(random_state)
    classes     = np.unique(y_train)
    layer_sizes = [X_train.shape[1]] + list(hidden_layers) + [len(classes)]

    # Random initialisation (symmetry breaking)
    eps = 0.12
    initial_params = np.concatenate([
        (np.random.rand(layer_sizes[i + 1], 1 + layer_sizes[i]) * 2 * eps - eps).ravel()
        for i in range(len(layer_sizes) - 1)
    ])

    res = optimize.minimize(
        fun     = nn_cost_function,
        x0      = initial_params,
        args    = (layer_sizes, X_train, y_train, alpha),
        method  = 'L-BFGS-B',
        jac     = True,
        options = {'maxiter': max_iter, 'disp': verbose},
    )

    return HandmadeMLP(res.x, layer_sizes, classes)