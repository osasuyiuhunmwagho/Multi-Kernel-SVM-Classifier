"""Kernels, synthetic datasets, and plotting helpers used by models.py.

Nothing in here is an estimator — it is the scaffolding that lets the estimators
be exercised: three kernel functions, two toy data generators with known ground
truth, and enough matplotlib to see what the models are doing.
"""

import os

import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ---------------------------------------------------------------------------
# Bias column and label encoding
# ---------------------------------------------------------------------------

def add_bias(X):
    """Prepend a column of ones so a linear model can learn an intercept."""
    n = X.shape[0]
    return np.concatenate((np.ones((n, 1)), X), axis=1)


def drop_bias(X):
    """Inverse of `add_bias`. Assumes the first column is the all-ones column."""
    return X[:, 1:]


def one_hot(y, n_classes):
    """Turn a 1-d array of integer labels into an n-by-n_classes indicator matrix."""
    y = y.astype(int).flatten()
    return np.eye(n_classes)[y].astype(float)


# ---------------------------------------------------------------------------
# Kernels
#
# All three take (X1: m-by-d, X2: n-by-d) and return an m-by-n Gram matrix, so
# they are interchangeable wherever a `kernel_func` argument is expected. Bind
# the extra hyperparameters with a lambda or functools.partial before passing
# them in.
# ---------------------------------------------------------------------------

def linear_kernel(X1, X2):
    """<x, x'> — kernel methods using this reduce to their linear counterparts."""
    return X1 @ X2.T


def poly_kernel(X1, X2, degree):
    """(<x, x'> + 1)^degree — spans all monomials up to the given degree."""
    return (X1 @ X2.T + 1) ** degree


def rbf_kernel(X1, X2, width):
    """exp(-||x - x'||^2 / 2 width^2). `width` sets how far influence reaches."""
    distances = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-distances / (2 * (width ** 2)))


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_dataset(n, gen_model, rand_seed=None):
    """Generate a labelled 2-d toy dataset.

    gen_model 1: four Gaussian blobs strung out along the diagonal, so the
                 classes sit in a line and heavily overlap — a hard case for a
                 1-d projection to preserve.
    gen_model 2: four Gaussian blobs at the corners of a square, cleanly
                 separable by a linear model.
    gen_model 3: two interleaving half-moons; linearly inseparable on purpose,
                 which is what makes it a useful test for the kernel methods.

    Returns (X, Y) with X already carrying its bias column and Y one-hot.
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    if gen_model == 3:
        X, y = make_moons(n)
        n_class = 2
    else:
        d = 2
        shift = 1.8
        n_class = 4

        X = []
        y = []
        m = n // 4
        class_label = 0
        for i in [-1, 1]:
            for j in [-1, 1]:
                if gen_model == 1:
                    # Each class is displaced further along (1, 1).
                    X.append(np.random.randn(m, d) + class_label * shift)
                elif gen_model == 2:
                    # One class per quadrant.
                    X.append(np.random.randn(m, d) + shift * np.array([[i, j]]))
                else:
                    raise ValueError(f"Unknown generative model: {gen_model}")
                y.append(np.ones((m, 1)) * class_label)
                class_label += 1
        X = np.vstack(X)
        y = np.vstack(y)

    return add_bias(X), one_hot(y, n_class)


def make_moons(n, noise=0.05):
    """Two interleaving half-circles, the standard non-linear separation test.

    Returns raw (X: n-by-2, y: n-by-1) without a bias column or one-hot encoding
    — `make_dataset` adds both.
    """
    n_samples_out = n // 2
    n_samples_in = n - n_samples_out

    # Upper moon: the top half of a unit circle.
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    # Lower moon: the same arc flipped and slid over so the two interlock.
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x),
         np.append(outer_circ_y, inner_circ_y)]
    ).T
    X += np.random.randn(*X.shape) * noise

    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp),
         np.ones(n_samples_in, dtype=np.intp)]
    )[:, None]
    return X, y


def load_digits(filename='digits.csv'):
    """Load the bundled handwritten-digit sample as a 200-by-784 matrix.

    Unlabelled 28x28 greyscale images, flattened row-major — the clustering and
    kernel-PCA demos use it as a stand-in for real high-dimensional data.
    """
    return np.loadtxt(os.path.join(DATA_DIR, filename), delimiter=',')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_points(X, Y):
    """Scatter a 2-d dataset, one marker/colour per class.

    X carries its bias column; Y is one-hot (or any cluster indicator matrix,
    which is how the k-means demos reuse this).
    """
    k = Y.shape[1]
    markers = ['o', '+', 'd', 'x', '^', 'v', 's']
    colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

    X = drop_bias(X)
    labels = Y.argmax(axis=1)
    for i in range(k):
        Xpart = X[labels == i]
        plt.scatter(Xpart[:, 0], Xpart[:, 1],
                    marker=markers[i],
                    color=colors[i],
                    label=f'class {i}')


def axis_limits(X, pad=0.1):
    """Bounding box of a 2-d (bias-free) point cloud, with a little breathing room."""
    x_min = np.amin(X[:, 0]) - pad
    x_max = np.amax(X[:, 0]) + pad
    y_min = np.amin(X[:, 1]) - pad
    y_max = np.amax(X[:, 1]) + pad
    return x_min, x_max, y_min, y_max


def plot_decision_regions(X, Y, W, predict, grid_step=0.01):
    """Overlay a classifier's decision regions on the training points.

    Works by classifying a dense grid over the data's bounding box and filling
    each cell with its predicted class — cheap, and it makes no assumption about
    the shape of the boundary.
    """
    plot_points(X, Y)

    x_min, x_max, y_min, y_max = axis_limits(drop_bias(X))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    grid = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()]  # bias column first
    labels = predict(grid, W).argmax(axis=1).reshape(xx.shape)

    # Two entries per class so adjacent contour bands share a colour.
    plt.contourf(xx, yy, labels,
                 colors=['r', 'r', 'b', 'b', 'g', 'g', 'y', 'y'],
                 alpha=0.3)
    plt.legend()
    plt.show()


def show_digit(x):
    """Render one flattened 784-vector as a 28x28 image."""
    plt.imshow(x.reshape((28, 28)), cmap='gray')
    plt.show()


def show_digits(X, n=20, title=None):
    """Render the first `n` rows of X as a grid of 28x28 images."""
    fig = plt.figure(figsize=(16, 6))
    if title:
        fig.suptitle(title)
    for i in range(min(n, len(X))):
        ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape((28, 28)), cmap='gray')
    plt.show()
