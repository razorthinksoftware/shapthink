# -*- coding: utf-8 -*-
"""
@created on: 2019-11-27,
@author: Himaprasoon,
@version: v0.0.1

Description:

Sphinx Documentation Status:

"""
import scipy.special
import itertools
from numpy.random import choice
import numpy as np


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 100
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


class KernelSHap:

    def __init__(self, num_features, predict_func, bayesian_network, column_mapping, label_name, reference=None,
                 causal_sampling=True):
        self.num_features = num_features
        self.predict_func = predict_func
        self.reference = reference if reference else np.zeros(num_features)
        self.bayesian_network = bayesian_network
        self.column_mapping = column_mapping
        self.label_name = label_name
        self.causal_sampling = causal_sampling

    def probability_func(self, s):
        if not s:
            return s
        selected_columns = [self.column_mapping[i] for i in s]
        beliefs = self.bayesian_network.predict_proba({i: "T" for i in selected_columns})
        for state, belief in zip(network.states, beliefs):
            if state.name == self.label_name:
                continue
            if hasattr(belief, "parameters"):
                # belief.parameters = [{"T": 0, "F": 1.0}]
                possible_choices = list(belief.parameters[0].keys())
                draw = choice(possible_choices, 1, p=[belief.parameters[0][i] for i in possible_choices])
                # print(state.name, draw)
                if draw == "T":
                    s.append(self.column_mapping.index(state.name))

        return list(sorted(s))

    def explain(self, data_to_explain, as_dict=True):
        X = np.zeros((2 ** self.num_features, self.num_features + 1))
        X[:, -1] = 1
        weights = np.zeros(2 ** self.num_features)
        V = np.zeros((2 ** self.num_features, self.num_features))
        for i in range(2 ** self.num_features):
            V[i, :] = self.reference
        for i, s in enumerate(powerset(range(self.num_features))):
            s = list(s)
            if self.causal_sampling:
                s = self.probability_func(s)
            # print(s,data_to_explain,data_to_explain[s])
            V[i, s] = data_to_explain[s]
            X[i, s] = 1
            weights[i] = shapley_kernel(self.num_features, len(s))
        y = self.predict_func(V)
        out = self.weighted_linear_reg(X, weights, y)
        if as_dict:
            return {i: j for i, j in zip(column_mapping, out[:-1])}, out
        return out

    @staticmethod
    def weighted_linear_reg(X, weights, y):
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


if __name__ == '__main__':
    import time
    from bayesian_lucas import network
    # print(network.predict_proba({"Genetics":"T"},max_iterations=100000))
    from model_test import predict, X_test
    import random

    # def predict_funct(X):
    #     return random.random()

    column_mapping = ['Smoking', 'Yellow_Fingers', 'Anxiety', 'Peer_Pressure', 'Genetics', 'Attention_Disorder',
                      'Born_an_Even_Day', 'Car_Accident', 'Fatigue', 'Allergy', 'Coughing']
    a = KernelSHap(num_features=11, predict_func=predict, bayesian_network=network, column_mapping=column_mapping,
                   label_name="Lung_cancer", causal_sampling=True)
    # start = time.time()
    # print(X_test)
    index = 0
    data = X_test.values[index]
    print(data)
    dic_out, out= a.explain(data,as_dict=True)
    # print(predict([data]))
    # print(time.time()-start)
    print(dic_out)
    import shap
    # shap.image_plot(out[0:-1],X_test.iloc[9, :])
    print(out)
    # print()
    shap.force_plot(out[-1], out[0:-1], X_test.iloc[index, :], matplotlib=True, out_names="Causal True")
