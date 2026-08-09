"""Runnable demos for everything in models.py.

Run the whole set:

    python demo.py

or a single one by name:

    python demo.py moons

Each demo either prints a table or opens a matplotlib window; close the window
to continue to the next one.
"""

import sys

import numpy as np
from matplotlib import pyplot as plt

import models
from utils import (make_dataset, load_digits, rbf_kernel, linear_kernel,
                   plot_points, plot_decision_regions, show_digits)


def demo_classifier():
    """Fit the softmax classifier on four blobs and draw its decision regions."""
    Xtrain, Ytrain = make_dataset(n=100, gen_model=1, rand_seed=0)

    W = models.fit_softmax_classifier(Xtrain, Ytrain)
    acc = models.accuracy(models.predict(Xtrain, W), Ytrain)
    print(f"[classifier] train accuracy: {acc:.3f}")

    plot_decision_regions(Xtrain, Ytrain, W, models.predict)


def demo_training_size():
    """Accuracy as a function of training-set size (rows: 16/32/64/128)."""
    train_acc, test_acc = models.training_size_sweep()
    print("[training size] rows = n_train (16, 32, 64, 128), cols = layout (1, 2)")
    print("train:\n", np.round(train_acc, 3))
    print("test:\n", np.round(test_acc, 3))


def demo_pca():
    """Accuracy after squeezing the data through 1 or 2 principal components."""
    train_acc, test_acc = models.pca_dimension_sweep()
    print("[pca] rows = n_components (1, 2), cols = layout (1, 2)")
    print("train:\n", np.round(train_acc, 3))
    print("test:\n", np.round(test_acc, 3))


def demo_kmeans():
    """Cluster the four-corner blobs with k = 3, ignoring the true labels."""
    Xtrain, _ = make_dataset(n=100, gen_model=2, rand_seed=0)

    Y, U, obj_val = models.kmeans_best_of(Xtrain, k=3, n_runs=20)
    print(f"[kmeans] best objective over 20 restarts: {obj_val:.4f}")

    plot_points(Xtrain, Y)
    plt.title('k-means, k = 3')
    plt.legend()
    plt.show()


def demo_elbow():
    """Objective vs. k on data that genuinely has four clusters."""
    Xtrain, _ = make_dataset(n=200, gen_model=2, rand_seed=0)

    k_candidates = list(range(2, 10))
    obj_vals = models.elbow_curve(Xtrain, k_candidates)
    for k, obj_val in zip(k_candidates, obj_vals):
        print(f"[elbow] k = {k}: {obj_val:.4f}")

    plt.plot(k_candidates, obj_vals, marker='o')
    plt.xlabel('k')
    plt.ylabel('objective')
    plt.title('Elbow curve (data has 4 true clusters)')
    plt.show()


def demo_moons():
    """The payoff case: kernel k-means separating two interleaving moons.

    Plain k-means can only cut the plane with a straight line, so it slices both
    moons in half. Kernel k-means with an RBF kernel recovers the two arcs
    exactly — but only from random restarts. Warm-starting it from the k-means
    answer instead returns that answer unchanged, because a k-means labelling is
    already a fixed point of the kernel update.
    """
    np.random.seed(0)
    X, Ytrue = make_dataset(n=100, gen_model=3)

    Y_plain, _, _ = models.kmeans(X, k=2)

    kernel_func = lambda X1, X2: rbf_kernel(X1, X2, width=0.25)
    Y_kernel, obj_val = models.kernel_kmeans_best_of(X, kernel_func, k=2)
    print(f"[moons] kernel k-means objective: {obj_val:.4f}")

    # Clusters carry no inherent order, so score both label-to-moon matchings
    # and keep the better one.
    truth = Ytrue.argmax(axis=1)
    for name, labels in [('k-means', Y_plain), ('kernel k-means', Y_kernel)]:
        agree = (labels.argmax(axis=1) == truth).mean()
        print(f"[moons] {name} agreement with true moons: {max(agree, 1 - agree):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, labels, title in [(axes[0], Y_plain, 'k-means'),
                              (axes[1], Y_kernel, 'kernel k-means (RBF, width 0.25)')]:
        plt.sca(ax)
        plot_points(X, labels)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def demo_digits():
    """Kernel PCA + clustering on the bundled 200-image digit sample."""
    X = load_digits()
    print(f"[digits] loaded {X.shape[0]} images of {X.shape[1]} pixels")

    # A linear kernel makes kernel PCA equivalent to ordinary PCA, which is a
    # useful sanity check that the centring and eigenvalue rescaling are right.
    A = models.kernel_pca(X, k=2, kernel_func=linear_kernel)
    Xproj = models.kernel_pca_project(X, X, linear_kernel, A)

    Y, U, obj_val = models.kmeans_best_of(Xproj, k=5, n_runs=20)
    print(f"[digits] k-means objective in 2-d kernel PCA space: {obj_val:.4f}")

    plt.figure(figsize=(6, 5))
    plot_points(np.c_[np.ones(len(Xproj)), Xproj], Y)  # plot_points wants a bias column
    plt.title('Digits projected onto 2 kernel-PCA components, clustered with k-means')
    plt.legend()
    plt.show()

    # Cluster means back in pixel space: each one should look like a blurry digit.
    centers = np.linalg.pinv(Y) @ X
    show_digits(centers, n=len(centers), title='k-means cluster means')


DEMOS = {
    'classifier': demo_classifier,
    'training-size': demo_training_size,
    'pca': demo_pca,
    'kmeans': demo_kmeans,
    'elbow': demo_elbow,
    'moons': demo_moons,
    'digits': demo_digits,
}


if __name__ == '__main__':
    requested = sys.argv[1:] or list(DEMOS)
    for name in requested:
        if name not in DEMOS:
            print(f"Unknown demo '{name}'. Choose from: {', '.join(DEMOS)}")
            sys.exit(1)
        DEMOS[name]()
