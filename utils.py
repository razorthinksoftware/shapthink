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


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 100
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


class KernelSHap:

    def __init__(self, num_features, predict_func, probability_func, bayesian_network, columns, reference=None):
        self.num_features = num_features
        self.predict_func = predict_func
        self.reference = reference if reference else np.zeros(num_features)
        self.probability_func = probability_func
        self.bayesian_network = bayesian_network
        self.columns = columns

    def explain(self, data_to_explain):
        X = np.zeros((2 ** self.num_features, self.num_features + 1))
        X[:, -1] = 1
        weights = np.zeros(2 ** self.num_features)
        V = np.zeros((2 ** self.num_features, self.num_features))
        for i in range(2 ** self.num_features):
            V[i, :] = self.reference
        for i, s in enumerate(powerset(range(self.num_features))):
            s = self.probability_func(list(s))
            V[i, s] = data_to_explain[s]
            X[i, s] = 1
            weights[i] = shapley_kernel(self.num_features, len(s))
        y = self.predict_func(V)
        return self.weighted_linear_reg(X, weights, y)

    @staticmethod
    def weighted_linear_reg(X, weights, y):
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


if __name__ == '__main__':
    from bayesian_lucas import network
    # print(network.predict_proba({"Genetics":"T"},max_iterations=100000))
    import random


    def predict_funct(X):
        return random.random()

    print([i.name for i in network.states])

    # KernelSHap(num_features=11, predict_func=predict_funct, bayesian_network, columns=[])
