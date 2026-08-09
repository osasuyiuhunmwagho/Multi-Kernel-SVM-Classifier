"""Linear and kernel methods written from scratch on top of NumPy/SciPy.

I wrote this to find out how much of "classical" machine learning you can get
working with nothing but matrix algebra and a general-purpose optimiser. No
scikit-learn anywhere: every estimator here is derived from its objective
function and solved directly.

Three families live in this module:

    * a multi-class softmax (multinomial logistic) classifier,
    * PCA and its kernelised counterpart,
    * k-means and kernel k-means.

Shape conventions used everywhere:

    X       n-by-d data matrix, one sample per row
    Y       n-by-k one-hot label matrix
    W       d-by-k weight matrix
    U       k-by-d matrix whose rows are directions (PCA) or centers (k-means)

Functions whose name mentions a bias term expect X to already carry a leading
column of ones; see `add_bias` in utils.py.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from utils import add_bias, drop_bias, make_dataset

# Fixed so the reported numbers in the README are reproducible.
RANDOM_SEED = 0


# ---------------------------------------------------------------------------
# Multi-class softmax classifier
# ---------------------------------------------------------------------------

def fit_softmax_classifier(X, Y):
    """Fit a linear multi-class classifier by minimising multinomial deviance.

    The objective is the average negative log-likelihood of the softmax model,

        (1/n) sum_i [ log(1_k^T exp(W^T x_i)) - y_i^T W^T x_i ]

    which is convex in W, so any starting point reaches the global optimum.

    X: n-by-d (bias column included), Y: n-by-k one-hot. Returns W: d-by-k.
    """
    n, d = X.shape
    k = Y.shape[1]

    def objective(w):
        # L-BFGS-B only understands flat parameter vectors, so unpack here.
        W = w.reshape(d, k)
        Z = X @ W  # n-by-k score matrix; row i holds (W^T x_i)^T

        # logsumexp rather than log(sum(exp(.))): the raw version overflows as
        # soon as the scores grow past ~700.
        lse = logsumexp(Z, axis=1)  # n-vector
        loss = (np.sum(lse) - np.sum(Y * Z)) / n

        # Analytic gradient (1/n) X^T (softmax(Z) - Y). Handing this to the
        # optimiser instead of letting it finite-difference is worth roughly an
        # order of magnitude in wall time on the larger runs.
        P = np.exp(Z - lse[:, None])  # row-wise softmax probabilities
        grad = X.T @ (P - Y) / n
        return loss, grad.ravel()

    w0 = np.zeros(d * k)
    result = minimize(objective, w0, jac=True, method='L-BFGS-B')
    return result.x.reshape(d, k)


def predict(Xtest, W):
    """Predict one-hot labels: each row is the argmax of the class scores.

    Xtest: m-by-d, W: d-by-k. Returns Yhat: m-by-k.
    """
    k = W.shape[1]
    labels = np.argmax(Xtest @ W, axis=1)
    return np.eye(k)[labels].astype(float)


def accuracy(Yhat, Y):
    """Fraction of rows where the predicted class matches the true class.

    Both arguments are m-by-k one-hot matrices.
    """
    predicted = np.argmax(Yhat, axis=1)
    truth = np.argmax(Y, axis=1)
    return float(np.mean(predicted == truth))


# ---------------------------------------------------------------------------
# Principal component analysis
# ---------------------------------------------------------------------------

def pca(X, k):
    """Return the top-k principal directions of X as the rows of a k-by-d matrix.

    Solved as an eigenproblem on the d-by-d scatter matrix rather than via SVD,
    which is the cheaper route whenever d is much smaller than n.

    X: n-by-d (no bias column), 1 <= k <= d.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    Xc = X - np.mean(X, axis=0)  # centre the data (broadcasts over rows)
    S = Xc.T @ Xc                # d-by-d scatter matrix

    # eigh returns eigenvalues in ascending order, so the top-k directions are
    # the *last* k columns; subset_by_index avoids computing the rest at all.
    _, eig_vecs = eigh(S, subset_by_index=[d - k, d - 1])
    return eig_vecs[:, ::-1].T   # flip to descending order, rows = directions


def pca_project(Xtest, mu, U):
    """Project data onto the principal directions: (Xtest - mu^T) U^T.

    mu must be the mean of the *training* set, otherwise train and test end up
    in different coordinate frames.

    Xtest: m-by-d, mu: length-d, U: k-by-d. Returns m-by-k.
    """
    mu = np.asarray(mu, dtype=float).reshape(1, -1)  # 1-by-d, for broadcasting
    return (np.asarray(Xtest, dtype=float) - mu) @ U.T


def kernel_pca(X, k, kernel_func):
    """Kernel PCA: top-k eigenvectors of the centred kernel matrix.

    Returns A: k-by-n, the (rescaled) dual coefficients. Feed it to
    `kernel_pca_project` together with the same training set and kernel.
    """
    X = np.asarray(X, dtype=float)  # float cast guards against integer overflow
    n = X.shape[0]                  # inside polynomial kernels
    K = kernel_func(X, X)

    # Centring in feature space:
    #   Kt = K - (1/n) 1 1^T K - (1/n) K 1 1^T + (1/n^2) 1 1^T K 1 1^T
    # which reduces to: subtract column means, subtract row means, add back the
    # grand mean (it gets removed twice).
    col_means = np.mean(K, axis=0, keepdims=True)  # 1-by-n
    row_means = np.mean(K, axis=1, keepdims=True)  # n-by-1
    Kt = K - col_means - row_means + np.mean(K)

    eig_vals, eig_vecs = eigh(Kt, subset_by_index=[n - k, n - 1])
    eig_vals = eig_vals[::-1]      # descending
    eig_vecs = eig_vecs[:, ::-1]

    # eigh hands back unit-norm alphas, but the direction they induce in feature
    # space, u = sum_i alpha_i phi(x_i), has squared norm alpha^T Kt alpha = lambda.
    # Rescaling by 1/sqrt(lambda) makes those directions unit-norm, which is what
    # makes kernel_pca with a linear kernel agree with pca exactly.
    eig_vals = np.maximum(eig_vals, 1e-12)  # zero/negative eigenvalues are round-off
    return (eig_vecs / np.sqrt(eig_vals)).T


def kernel_pca_project(Xtest, Xtrain, kernel_func, A):
    """Project test points onto kernel principal directions: Kt_test,train A^T.

    Xtest: m-by-d, Xtrain: n-by-d, A: k-by-n. Returns m-by-k.
    """
    Xtest = np.asarray(Xtest, dtype=float)
    Xtrain = np.asarray(Xtrain, dtype=float)

    K_tr = kernel_func(Xtrain, Xtrain)   # n-by-n
    K_te = kernel_func(Xtest, Xtrain)    # m-by-n

    # Same centring as above, but the means always come from the training block
    # so that a test point is measured against the training feature-space mean:
    #   Kt = K_te - (1/n) 1_{m,n} K_tr - (1/n) K_te 1_{n,n}
    #             + (1/n^2) 1_{m,n} K_tr 1_{n,n}
    tr_col_means = np.mean(K_tr, axis=0, keepdims=True)  # 1-by-n
    te_row_means = np.mean(K_te, axis=1, keepdims=True)  # m-by-1
    Kt = K_te - tr_col_means - te_row_means + np.mean(K_tr)

    return Kt @ A.T


def training_size_sweep(n_runs=10, n_test=1000, n_train_list=(16, 32, 64, 128)):
    """How the softmax classifier's accuracy scales with the training-set size.

    The gap between the train and test columns is the interesting part: at
    n_train = 16 the model fits its training set almost perfectly and
    generalises poorly, and the two converge as n grows.

    Returns (train_acc, test_acc), each len(n_train_list)-by-len(gen_model_list).
    """
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(n_train_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(n_train_list), len(gen_model_list), n_runs])

    np.random.seed(RANDOM_SEED)
    for r in range(n_runs):
        for i, n_train in enumerate(n_train_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = make_dataset(n=n_train, gen_model=gen_model)
                Xtest, Ytest = make_dataset(n=n_test, gen_model=gen_model)

                W = fit_softmax_classifier(Xtrain, Ytrain)
                train_acc[i, j, r] = accuracy(predict(Xtrain, W), Ytrain)
                test_acc[i, j, r] = accuracy(predict(Xtest, W), Ytest)

    return np.mean(train_acc, axis=2), np.mean(test_acc, axis=2)


def pca_dimension_sweep(n_runs=100, n_train=128, n_test=1000):
    """How much accuracy survives when the classifier only sees k PCA components?

    Runs the softmax classifier on 1- and 2-dimensional projections of both
    synthetic layouts and averages over `n_runs` resamples.

    Returns (train_acc, test_acc), each len(dim_list)-by-len(gen_model_list).
    """
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])

    np.random.seed(RANDOM_SEED)
    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = make_dataset(n=n_train, gen_model=gen_model)
                Xtest, Ytest = make_dataset(n=n_test, gen_model=gen_model)

                # Strip the bias column before fitting PCA; it is constant, so it
                # would only contribute a zero-variance direction, and it has to
                # be re-added afterwards anyway for the classifier.
                Xtrain, Xtest = drop_bias(Xtrain), drop_bias(Xtest)

                U = pca(Xtrain, k)
                mu = np.mean(Xtrain, axis=0)  # training mean used for both splits
                Xtrain_proj = add_bias(pca_project(Xtrain, mu, U))
                Xtest_proj = add_bias(pca_project(Xtest, mu, U))

                W = fit_softmax_classifier(Xtrain_proj, Ytrain)
                train_acc[i, j, r] = accuracy(predict(Xtrain_proj, W), Ytrain)
                test_acc[i, j, r] = accuracy(predict(Xtest_proj, W), Ytest)

    return np.mean(train_acc, axis=2), np.mean(test_acc, axis=2)


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------

def kmeans(X, k, max_iter=1000):
    """Lloyd's algorithm.

    Returns (Y, U, obj_val): the n-by-k assignment matrix, the k-by-d centers,
    and the average half squared distance to the assigned center.

    Deliberately unseeded — `kmeans_best_of` relies on repeated calls landing in
    different local minima.
    """
    n, d = X.shape
    assert max_iter > 0 and k < n
    X = np.asarray(X, dtype=float)

    # Forgy initialisation: k distinct data points serve as the first centers.
    U = X[np.random.choice(n, k, replace=False), :]

    I_k = np.eye(k)
    D = cdist(X, U, 'sqeuclidean')
    for _ in range(max_iter):
        D = cdist(X, U, 'sqeuclidean')  # n-by-k pairwise distances
        Y = I_k[np.argmin(D, axis=1)]   # assign every point to its closest center
        old_U = U
        # The optimal centers given Y solve a least-squares problem whose answer
        # is U = (Y^T Y)^-1 Y^T X. pinv covers the case of an empty cluster,
        # where Y^T Y is singular.
        U = np.linalg.pinv(Y) @ X
        if np.allclose(old_U, U):
            break

    # Score against the final centers, not the ones D was last computed from.
    D = cdist(X, U, 'sqeuclidean')
    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, U, obj_val


def kmeans_best_of(X, k, n_runs=100):
    """Restart k-means `n_runs` times and keep the lowest-objective solution.

    Lloyd's algorithm only guarantees a local minimum, and with a bad Forgy draw
    the gap to the best run is large enough to matter.
    """
    best_obj_val = float('inf')
    best_Y = None
    best_U = None
    for _ in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        if obj_val < best_obj_val:
            best_Y, best_U, best_obj_val = Y, U, obj_val
    return best_Y, best_U, best_obj_val


def elbow_curve(X, k_candidates=(2, 3, 4, 5, 6, 7, 8, 9)):
    """Best objective value reached for each candidate k.

    The objective decreases monotonically in k, so the useful signal is the
    kink in the curve rather than the minimum.
    """
    return [kmeans_best_of(X, k)[2] for k in k_candidates]


def kernel_kmeans(X, kernel_func, k, init_Y=None, max_iter=1000):
    """k-means run in the feature space induced by `kernel_func`.

    Centers are never formed explicitly — they only exist as averages of feature
    vectors — so the algorithm works entirely with distances expanded through
    the kernel matrix. That is what lets it separate clusters no hyperplane
    through the input space could (two interleaving moons, for instance).

    The result is very sensitive to `init_Y`, and in a way that is easy to get
    backwards: seeding with a plain k-means labelling looks like a smart warm
    start but is actually a trap, because that labelling is already a fixed
    point of the update below and the algorithm terminates on it immediately.
    Random restarts scored by objective (`kernel_kmeans_best_of`) do far better.

    Returns (Y, obj_val).
    """
    n, d = X.shape
    assert max_iter > 0 and k < n
    X = np.asarray(X, dtype=float)
    K = kernel_func(X, X)

    if init_Y is None:
        init_Y = np.eye(k)[np.random.randint(0, k, n)]
    Y = np.asarray(init_Y, dtype=float)

    I_k = np.eye(k)
    D = None
    for _ in range(max_iter):
        # Squared distance from every point to every cluster mean, expanded as
        # ||phi(x) - m_j||^2 = K_ii - 2 (K Y^+T)_ij + (Y^+ K Y^+T)_jj:
        Y_pinv = np.linalg.pinv(Y)          # k-by-n
        M = K @ Y_pinv.T                    # n-by-k, the cross term
        centre_norms = np.diag(Y_pinv @ M)  # k-vector, ||mean of cluster j||^2
        D = np.diag(K)[:, None] + centre_norms[None, :] - 2.0 * M

        old_Y = Y
        Y = I_k[np.argmin(D, axis=1)]
        if np.allclose(old_Y, Y):
            break

    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, obj_val


def kernel_kmeans_best_of(X, kernel_func, k, n_runs=200):
    """Restart kernel k-means from random assignments and keep the best.

    Restarts matter far more here than for plain k-means. On the two-moons set
    the default of 200 recovers the true arcs every time, 100 manages it about
    two thirds of the time, and 50 essentially never does — the correct solution
    really is the lowest-objective one, it is just ringed by local minima that a
    single random start almost always falls into instead.

    Returns (Y, obj_val).
    """
    best_Y = None
    best_obj_val = float('inf')
    for _ in range(n_runs):
        Y, obj_val = kernel_kmeans(X, kernel_func, k, init_Y=None)
        if obj_val < best_obj_val:
            best_Y, best_obj_val = Y, obj_val
    return best_Y, best_obj_val
