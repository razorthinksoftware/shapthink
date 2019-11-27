# -*- coding: utf-8 -*-
"""
@created on: 2019-11-27,
@author: Himaprasoon,
@version: v0.0.1

Description:

Sphinx Documentation Status:

"""
import scipy.special
import numpy as np
import itertools

import numpy as np


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
