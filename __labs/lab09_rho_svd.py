#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 09/07/2021

"""

import numpy
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

"""
References: 
https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables/313138#313138
https://stats.stackexchange.com/questions/444039/whuber-s-generation-of-a-random-variable-with-fixed-covariance-structure
"""

"""
* X is the treatment variable 
* Y is the response variable 
* W is a confounder between X and Y

* X is fixed 
* Y will be generated given a correlation coefficient with X
* W will be generated given the correlation coeffs with X and Y.
"""

numpy.random.seed(123)


def get_response_var(X, rho, check=False):
    n_ = len(X)

    Y_temp = numpy.random.randn(n_).reshape(-1, 1)

    X_perp = OLS(Y_temp, add_constant(X)).fit().resid

    Y_ = rho_XY * X_perp.std() * X + numpy.sqrt(1 - rho_XY ** 2) * X.std() * X_perp

    Y_ /= norm(Y_)

    if check:
        print('rho_xy =', pearsonr(X, Y_)[0])

    return Y_


def get_confounder_var(X_, Y_, rho, check=False):
    X_rmse = mean_squared_error(X_, [X_.mean()] * n, squared=False)
    Y_rmse = mean_squared_error(Y_, [Y_.mean()] * n, squared=False)

    X_scale = X_ / X_rmse
    Y_scale = Y_ / Y_rmse

    XY = numpy.column_stack((X_scale, Y_scale))

    W_temp = numpy.random.randn(n).reshape(-1, 1)

    resid = OLS(W_temp, add_constant(XY)).fit().resid

    U, D, V = svd(XY, full_matrices=False)

    threshold = 1e-12

    D = [1 / d if d > threshold else 0 for d in D]

    D_pinv = numpy.diag(D)

    XY_dual = n * U @ D_pinv @ V.T

    sigma2 = (1 - rho.T @ numpy.cov(XY_dual, rowvar=False) @ rho) / resid.var()

    W_ = (XY_dual @ rho + numpy.sqrt(sigma2) * resid.reshape(-1, 1))

    if check:
        print('Sanity check for identity matrix:\n', XY_dual.T @ XY / n)

        # TODO: pretty table print with target rho and actual rho
        print(pearsonr(X_, W_.squeeze())[0])
        print(pearsonr(Y_, W_.squeeze())[0])

    return W_


n = 100

X = numpy.linspace(-1, 1, n)

rho_XY = 0.6

Y = get_response_var(X, rho_XY, check=True)

rho_W_XY = numpy.array([0.8, 0.2]).reshape(-1, 1)

W = get_confounder_var(X, Y, rho_W_XY, check=True)





