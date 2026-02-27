# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.svm import LinearSVC

#from disentanglementmetrics.src.utils import get_bin_index

def get_bin_index(x, nb_bins):
    ''' Discretize input variable
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)


def _value_counts(values):
    if values is None:
        return {}
    uniques, counts = np.unique(values, return_counts=True)
    return {float(u): int(c) for u, c in zip(uniques, counts)}
    
    
def sap(factors, codes, continuous_factors=True, nb_bins=10, regression=True):
    ''' SAP metric from A. Kumar, P. Sattigeri, and A. Balakrishnan,
        “Variational inference of disentangled latent concepts from unlabeledobservations,”
        in ICLR, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param regression:                      True:   compute score using regression algorithms
                                            False:  compute score using classification algorithms
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # perform regression
    if regression:
        assert(continuous_factors), f'Cannot perform SAP regression with discrete factors.'
        return _sap_regression(factors, codes, nb_factors, nb_codes)  
    
    # perform classification
    else:
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(factors)  # normalize in [0, 1] all columns
            factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
        
        # normalize in [0, 1] all columns
        codes = minmax_scale(codes)
        
        # compute score using classification algorithms
        return _sap_classification(factors, codes, nb_factors, nb_codes)


def _sap_regression_matrix(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using regression algorithms
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute R2 score matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # train a linear regressor
            regr = LinearRegression()

            # train the model using the training sets
            regr.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = regr.predict(codes[:, c].reshape(-1, 1))

            # compute R2 score
            r2 = r2_score(factors[:, f], y_pred)
            s_matrix[f, c] = max(0, r2)

    return s_matrix


def _sap_regression(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using regression algorithms

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    s_matrix = _sap_regression_matrix(factors, codes, nb_factors, nb_codes)

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]
    
    # compute the mean gap
    sap_score = sum_gap / nb_factors
    
    return sap_score


def _sap_classification_matrix(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using classification algorithms
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute accuracy matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # find the optimal number of splits
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                # perform cross validation on the tree classifiers
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                scores = cross_val_score(
                    clf,
                    codes[:, c].reshape(-1, 1),
                    factors[:, f].reshape(-1, 1),
                    cv=5,
                )
                scores = scores.mean()

                if scores > best_score:
                    best_score = scores
                    best_sp = sp

            # train the model using the best performing parameter
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            clf.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = clf.predict(codes[:, c].reshape(-1, 1))

            # compute accuracy
            s_matrix[f, c] = accuracy_score(y_pred, factors[:, f])

    return s_matrix


def _sap_classification_predictions(
    factors, codes, nb_factors, nb_codes, pred_sample_n=0
):
    ''' Return prediction summaries for SAP classification. '''
    pred_info = [[None for _ in range(nb_codes)] for _ in range(nb_factors)]
    for f in range(nb_factors):
        for c in range(nb_codes):
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                scores = cross_val_score(
                    clf,
                    codes[:, c].reshape(-1, 1),
                    factors[:, f].reshape(-1, 1),
                    cv=5,
                )
                scores = scores.mean()

                if scores > best_score:
                    best_score = scores
                    best_sp = sp

            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            clf.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))
            y_pred = clf.predict(codes[:, c].reshape(-1, 1))

            info = {
                "pred_counts": _value_counts(y_pred),
                "true_counts": _value_counts(factors[:, f]),
            }
            if pred_sample_n and pred_sample_n > 0:
                info["pred_sample"] = y_pred[:pred_sample_n].tolist()
            pred_info[f][c] = info

    return pred_info


def _sap_classification(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using classification algorithms

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    s_matrix = _sap_classification_matrix(factors, codes, nb_factors, nb_codes)

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]
    
    # compute the mean gap
    sap_score = sum_gap / nb_factors
    
    return sap_score


def sap_score_matrix(factors, codes, continuous_factors=True, nb_bins=10, regression=True):
    ''' Return SAP score matrix for per-latent reporting.

    :param factors:                         dataset of factors
    :param codes:                           latent codes associated to the dataset of factors
    :param continuous_factors:              True if factors are continuous
    :param nb_bins:                         number of bins to use for discretization
    :param regression:                      True for regression, False for classification
    '''
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    if regression:
        assert continuous_factors, "Cannot perform SAP regression with discrete factors."
        return _sap_regression_matrix(factors, codes, nb_factors, nb_codes)

    if continuous_factors:
        factors = minmax_scale(factors)
        factors = get_bin_index(factors, nb_bins)

    codes = minmax_scale(codes)
    return _sap_classification_matrix(factors, codes, nb_factors, nb_codes)


def sap_classification_predictions(
    factors, codes, continuous_factors=True, nb_bins=10, pred_sample_n=0
):
    ''' Return prediction summaries for SAP classification (Kumar et al.). '''
    factors = np.asarray(factors)
    codes = np.asarray(codes)
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if codes.ndim != 2:
        raise ValueError("codes must be 2D [N, D]")

    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    if continuous_factors:
        factors = minmax_scale(factors)
        factors = get_bin_index(factors, nb_bins)

    codes = minmax_scale(codes)
    return _sap_classification_predictions(
        factors, codes, nb_factors, nb_codes, pred_sample_n=pred_sample_n
    )


def sap_classification_holdout_predictions(
    factors,
    codes,
    continuous_factors=True,
    nb_bins=10,
    train_frac=0.8,
    random_state=0,
    pred_sample_n=0,
):
    ''' Return train/test prediction summaries for SAP classification. '''
    factors = np.asarray(factors)
    codes = np.asarray(codes)
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if codes.ndim != 2:
        raise ValueError("codes must be 2D [N, D]")

    n_samples = factors.shape[0]
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    train_acc = np.full((nb_factors, nb_codes), np.nan, dtype=float)
    test_acc = np.full((nb_factors, nb_codes), np.nan, dtype=float)
    pred_info = [[None for _ in range(nb_codes)] for _ in range(nb_factors)]

    if n_samples < 4:
        return train_acc, test_acc, pred_info

    if continuous_factors:
        factors = minmax_scale(factors)
        factors = get_bin_index(factors, nb_bins)

    codes = minmax_scale(codes)

    for f in range(nb_factors):
        y = factors[:, f].reshape(-1)
        for c in range(nb_codes):
            x = codes[:, c].reshape(-1, 1)
            mask = np.isfinite(y) & np.isfinite(x).reshape(-1)
            y_valid = y[mask]
            x_valid = x[mask]

            if y_valid.size < 4:
                continue
            if np.unique(y_valid).size < 2:
                continue

            test_size = max(1, int(round((1.0 - train_frac) * y_valid.size)))
            train_size = y_valid.size - test_size
            if train_size < 2:
                continue

            stratify = y_valid if np.unique(y_valid).size > 1 else None
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_valid,
                    y_valid,
                    test_size=test_size,
                    train_size=train_size,
                    random_state=random_state,
                    stratify=stratify,
                )
            except ValueError:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_valid,
                    y_valid,
                    test_size=test_size,
                    train_size=train_size,
                    random_state=random_state,
                    stratify=None,
                )

            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                try:
                    scores = cross_val_score(
                        clf,
                        x_train,
                        y_train,
                        cv=5,
                    )
                    scores = scores.mean()
                except ValueError:
                    scores = 0

                if scores > best_score:
                    best_score = scores
                    best_sp = sp

            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            clf.fit(x_train, y_train)

            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            train_acc[f, c] = accuracy_score(y_train, y_pred_train)
            test_acc[f, c] = accuracy_score(y_test, y_pred_test)

            info = {
                "train_pred_counts": _value_counts(y_pred_train),
                "train_true_counts": _value_counts(y_train),
                "test_pred_counts": _value_counts(y_pred_test),
                "test_true_counts": _value_counts(y_test),
            }
            if pred_sample_n and pred_sample_n > 0:
                info["train_pred_sample"] = y_pred_train[:pred_sample_n].tolist()
                info["test_pred_sample"] = y_pred_test[:pred_sample_n].tolist()
            pred_info[f][c] = info

    return train_acc, test_acc, pred_info


def sap_regression_predictions(factors, codes, pred_sample_n=0):
    ''' Return prediction summaries for SAP regression (Kumar et al.). '''
    factors = np.asarray(factors)
    codes = np.asarray(codes)
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if codes.ndim != 2:
        raise ValueError("codes must be 2D [N, D]")

    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    pred_info = [[None for _ in range(nb_codes)] for _ in range(nb_factors)]
    for f in range(nb_factors):
        for c in range(nb_codes):
            regr = LinearRegression()
            regr.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))
            y_pred = regr.predict(codes[:, c].reshape(-1, 1)).reshape(-1)
            info = {
                "pred_mean": float(np.mean(y_pred)) if y_pred.size else float("nan"),
                "pred_std": float(np.std(y_pred)) if y_pred.size else float("nan"),
            }
            if pred_sample_n and pred_sample_n > 0:
                info["pred_sample"] = y_pred[:pred_sample_n].tolist()
            pred_info[f][c] = info
    return pred_info


def sap_binary_classification_locatello(
    factors,
    codes,
    train_frac=0.8,
    C=0.01,
    random_state=0,
    return_predictions=False,
    pred_sample_n=0,
):
    ''' SAP binary classification using Locatello-style protocol.

    For each factor and each single latent dimension, train a linear SVM
    and store test-set prediction error. SAP is the mean (over factors)
    of the gap between the lowest and second-lowest errors.
    '''
    factors = np.asarray(factors)
    codes = np.asarray(codes)
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if codes.ndim != 2:
        raise ValueError("codes must be 2D [N, D]")

    n_samples = factors.shape[0]
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    error_matrix = np.full((nb_factors, nb_codes), np.nan, dtype=float)
    pred_info = None
    if return_predictions:
        pred_info = [[None for _ in range(nb_codes)] for _ in range(nb_factors)]

    if n_samples < 4:
        return float("nan"), error_matrix

    test_size = max(1, int(round((1.0 - train_frac) * n_samples)))
    train_size = n_samples - test_size
    if train_size < 2:
        return float("nan"), error_matrix

    for f in range(nb_factors):
        y = factors[:, f].reshape(-1)
        for c in range(nb_codes):
            x = codes[:, c].reshape(-1, 1)
            mask = np.isfinite(y) & np.isfinite(x).reshape(-1)
            y_valid = y[mask]
            x_valid = x[mask]

            if y_valid.size < 4:
                continue
            if np.unique(y_valid).size < 2:
                continue

            stratify = y_valid if np.unique(y_valid).size > 1 else None
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_valid,
                    y_valid,
                    test_size=test_size,
                    train_size=train_size,
                    random_state=random_state,
                    stratify=stratify,
                )
            except ValueError:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_valid,
                    y_valid,
                    test_size=test_size,
                    train_size=train_size,
                    random_state=random_state,
                    stratify=None,
                )

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            clf = LinearSVC(C=C, max_iter=5000)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            error = 1.0 - accuracy_score(y_test, y_pred)
            error_matrix[f, c] = error
            if return_predictions:
                info = {
                    "pred_counts": _value_counts(y_pred),
                    "true_counts": _value_counts(y_test),
                }
                if pred_sample_n and pred_sample_n > 0:
                    info["pred_sample"] = y_pred[:pred_sample_n].tolist()
                    info["true_sample"] = y_test[:pred_sample_n].tolist()
                pred_info[f][c] = info

    gaps = []
    for f in range(nb_factors):
        vals = error_matrix[f, :]
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        vals_sorted = np.sort(vals)
        gaps.append(vals_sorted[1] - vals_sorted[0])

    sap_score = float(np.mean(gaps)) if gaps else float("nan")
    if return_predictions:
        return sap_score, error_matrix, pred_info
    return sap_score, error_matrix
