"""Small, dependency-free reference math for cache-mapping smoke checks.

Real checkpoint calibration uses the tensor backend. These routines verify the
affine/RoPE conventions on tiny matrices, not model-quality claims.
"""
import math


def solve(a, b):
    n = len(a)
    if not n or any(len(r) != n for r in a) or len(b) != n or not b[0] or len({len(r) for r in b}) != 1:
        raise ValueError("Invalid linear system")
    if not all(math.isfinite(v) for r in [*a, *b] for v in r):
        raise ValueError("Nonfinite linear system")
    width = len(b[0])
    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for row in range(n):
            if row != col:
                scale = aug[row][col]
                aug[row] = [x - scale*y for x, y in zip(aug[row], aug[col])]
    return [row[n:n+width] for row in aug]


def fit_affine(x, y, alpha=.01):
    if len(x) != len(y) or len(x) < 2 or not x[0] or not y[0] or alpha < 0:
        raise ValueError("Invalid ridge inputs")
    dx, dy, n = len(x[0]), len(y[0]), len(x)
    if any(len(r) != dx for r in x) or any(len(r) != dy for r in y):
        raise ValueError("Ragged ridge inputs")
    mx = [sum(row[j] for row in x)/n for j in range(dx)]
    my = [sum(row[j] for row in y)/n for j in range(dy)]
    xx = [[sum((r[i]-mx[i])*(r[j]-mx[j]) for r in x)/n + (alpha if i == j else 0) for j in range(dx)] for i in range(dx)]
    xy = [[sum((a[i]-mx[i])*(b[j]-my[j]) for a, b in zip(x, y))/n for j in range(dy)] for i in range(dx)]
    weight = solve(xx, xy)
    bias = [my[j] - sum(mx[i]*weight[i][j] for i in range(dx)) for j in range(dy)]
    return weight, bias


def affine(vector, weight, bias):
    return [sum(vector[i]*weight[i][j] for i in range(len(vector))) + bias[j] for j in range(len(bias))]


def rotate(vector, position, theta=1_000_000., inverse=False):
    dim = len(vector)
    if dim == 0 or dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    half = dim // 2
    output = [0.] * dim
    for i in range(half):
        angle = position / theta**(2*i/dim) * (-1 if inverse else 1)
        c, s = math.cos(angle), math.sin(angle)
        output[i] = vector[i]*c - vector[i+half]*s
        output[i+half] = vector[i]*s + vector[i+half]*c
    return output
