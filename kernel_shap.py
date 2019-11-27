import scipy.special
import numpy as np
import itertools


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 100
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


def f(X):
    np.random.seed(0)
    beta = np.random.rand(X.shape[-1])
    return np.dot(X, beta) + 10


def kernel_shap(f, x, reference, M):
    X = np.zeros((2 ** M, M + 1))
    X[:, -1] = 1
    weights = np.zeros(2 ** M)
    V = np.zeros((2 ** M, M))
    for i in range(2 ** M):
        V[i, :] = reference
    ws = {}
    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        V[i, s] = x[s]
        X[i, s] = 1
        ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(M, len(s))
        weights[i] = shapley_kernel(M, len(s))
    y = f(V)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


M = 2
np.random.seed(0)
x = np.random.randn(M)
reference = np.zeros(M)

print("  reference =", reference)
print("          x =", x)
print("       f(x) =", f(x))
phi = kernel_shap(f, x, reference, M)
base_value = phi[-1]
shap_values = phi[:-1]

print("shap_values =", shap_values)
print(sum([i for i in shap_values]))
print(f(x) - 10)
# exit()
print(" base_value =", base_value)
print("   sum(phi) =", np.sum(phi))

print("__SHAP__" * 10)
import shap

explainer = shap.KernelExplainer(f, np.reshape(reference, (1, len(reference))))
shap_values = explainer.shap_values(x)
print("shap_values =", shap_values)
print("base value =", explainer.expected_value)
