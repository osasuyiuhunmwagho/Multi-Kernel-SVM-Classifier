# Kernel Methods from Scratch

Classical linear and kernel machine learning implemented directly from their
objective functions, using nothing but NumPy and SciPy. No scikit-learn — the
point was to see how much of this you can build out of plain matrix algebra and
a general-purpose optimiser, and to find out where the textbook version of each
algorithm quietly disagrees with what actually happens when you run it.

## What's implemented

| | Method | Notes |
|---|---|---|
| Classification | Multi-class softmax (multinomial logistic) regression | Convex objective solved with L-BFGS-B and an analytic gradient |
| Dimensionality reduction | PCA | Eigendecomposition of the scatter matrix |
| | Kernel PCA | Feature-space centring + eigenvalue rescaling, so a linear kernel reproduces plain PCA exactly |
| Clustering | k-means | Lloyd's algorithm, Forgy init, restart-and-keep-best |
| | Kernel k-means | Centers never formed explicitly; distances expanded through the Gram matrix |

Kernels available: linear, polynomial, and RBF. They share one signature
(`kernel_func(X1, X2) -> Gram matrix`) so they drop into any of the kernel
methods interchangeably.

## Running it

```bash
pip install -r requirements.txt
python demo.py            # run every demo
python demo.py moons      # or just one
```

Demos: `classifier`, `training-size`, `pca`, `kmeans`, `elbow`, `moons`, `digits`.

## Layout

```
models.py    the estimators and the two experiment sweeps
utils.py     kernels, synthetic data generators, plotting
demo.py      runnable demos, one function each
data/        200 unlabelled 28x28 handwritten digits (200 x 784 CSV)
```

## Things worth pointing out

**Kernel k-means lives or dies on its initialisation, in a counterintuitive
way.** Warm-starting it from a plain k-means labelling looks obviously smart and
is in fact useless: that labelling is already a fixed point of the kernel
update, so the algorithm converges on iteration one and hands the straight-line
answer straight back. On two interleaving moons it sits at 74% agreement with
the true arcs and never moves.

Random restarts scored by objective get it right instead, but need more of them
than seemed reasonable. The correct clustering genuinely *is* the lowest-objective
one — it is just ringed by local minima:

| Restarts | Agreement with true moons |
|---|---|
| 50 | ~0.87 |
| 100 | ~0.95 |
| 200 | 1.00 (every seed tested) |

**How much PCA can throw away depends entirely on how the classes are arranged.**
Averaged over 100 runs, projecting to one dimension and classifying:

| Components | Blobs along a diagonal | Blobs at square corners |
|---|---|---|
| 1 | 0.843 | 0.636 |
| 2 | 0.836 | 0.917 |

When the classes are strung out along a line, one component keeps nearly
everything — the second dimension was never carrying class information. When
they sit at the corners of a square, collapsing to one axis merges two pairs of
classes and accuracy falls off a cliff. PCA optimises for variance, and variance
is not the same thing as class separation.

**The elbow is only obvious when it's real.** On data with four true clusters
the objective drops 1.54 → 0.87 going from k=3 to k=4, then flattens into a slow
decline (0.74, 0.65, 0.56, ...). The kink is the signal; the minimum is not,
since the objective decreases monotonically in k by construction.

## Implementation notes

A few places where the naive version doesn't survive contact with real numbers:

- The softmax loss goes through `logsumexp` rather than `log(sum(exp(.)))`,
  which overflows once scores pass ~700.
- Kernel PCA rescales each eigenvector by `1/sqrt(lambda)`. `eigh` returns
  unit-norm coefficient vectors, but the direction they induce in feature space
  has squared norm `lambda`, so without the rescaling a linear kernel does *not*
  reproduce PCA. That equivalence is used as a correctness check.
- Both k-means variants use `pinv` for the center update, which keeps an empty
  cluster from producing a singular matrix.
