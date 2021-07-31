#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 02/07/2021

"""

import numpy.random

from scipy.linalg import svd, norm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def scale_vector(arr):
    rmse = mean_squared_error(arr, [arr.mean()] * len(arr), squared=False)

    return arr / rmse


def get_pearsonr(x, y):
    return pearsonr(x, y)[0]


def correlated_vector(arr, rho, check=False):
    assert len(arr) == len(rho), "Number of arrays and corr coeffs given does not match"

    if isinstance(arr, list):
        arr = numpy.array(arr)

    if isinstance(rho, list):
        rho = numpy.array(rho).reshape(-1, 1)

    scaled_vecs = numpy.apply_along_axis(scale_vector, axis=1, arr=arr).T

    n = len(scaled_vecs)

    X_temp = numpy.random.randn(n).reshape(-1, 1)

    resid = OLS(X_temp, add_constant(scaled_vecs)).fit().resid

    U, D, V = svd(scaled_vecs, full_matrices=False)

    threshold = 1e-12

    D = [1 / d if d > threshold else 0 for d in D]

    D_pinv = numpy.diag(D)

    dual_basis = n * U @ D_pinv @ V.T

    sigma2 = (1 - rho.T @ numpy.cov(dual_basis, rowvar=False) @ rho) / resid.var()

    rho0 = get_pearsonr(arr[0], arr[1])

    if sigma2 < 0:
        det = 1 + 2 * rho0 * rho[0] * rho[1] - (rho0 ** 2 + rho[0] ** 2 + rho[1] ** 2)

        raise ValueError(f'determinant = {det[0]}: impossible correlation')

    I = dual_basis.T @ scaled_vecs / n

    if not numpy.isclose(I, numpy.eye(len(arr))).all():
        if I[0][1] < 0:
            dual_basis[:, 0] *= -1
        if I[1][0] < 0:
            dual_basis[:, 1] *= -1

        dual_basis = swap_arr_columns(dual_basis.copy())

        print('HERE')

    X = dual_basis @ rho + numpy.sqrt(sigma2) * resid.reshape(-1, 1)

    try:

        X = (X / norm(X)).squeeze()

    except:
        I = dual_basis.T @ scaled_vecs / n
        print('a')

    if check:

        # print(f'{sigma2 = }')
        print('Sanity check for identity matrix:\n', dual_basis.T @ scaled_vecs / n)

        # TODO: pretty table print with target rho and actual rho
        corrcoefs = numpy.apply_along_axis(get_pearsonr, axis=0, arr=scaled_vecs, y=X)

        for c in corrcoefs:
            print(c)

    return X


def swap_arr_columns(arr):
    arr_copy = arr.copy()

    arr_copy[:, 0] = arr[:, 1]
    arr_copy[:, 1] = arr[:, 0]

    return arr_copy


def correlated_vector_simple(arr, rho, check=False):
    if isinstance(rho, list):
        rho = rho[0]

    n_ = len(arr)

    X_temp = numpy.random.randn(n_).reshape(-1, 1)

    W_perp = OLS(X_temp, add_constant(arr)).fit().resid

    X_ = rho * W_perp.std() * arr + numpy.sqrt(1 - rho ** 2) * arr.std() * W_perp
    # X_ = rho * W_perp.std() * arr + numpy.sqrt(1 - rho ** 2) * arr.std() * W_perp

    X_ /= norm(X_)

    if check:
        print('rho_xy =', pearsonr(arr, X_)[0])

    return X_


# numpy.random.seed(42)

n = 10_000

pt = 0.25
p0 = -0.3
alpha_x = 1

cov = [[1, pt, 0], [pt, 1, 0], [0, 0, 1]]

samples = numpy.random.multivariate_normal(mean=[0, 0, 0], cov=cov, size=n).T
X_true = samples[0]
Y_true = samples[1]
W = samples[2]

bx = alpha_x * W.std() / X_true.std()

a = (p0 ** 2 - 1) * bx ** 2 + p0 ** 2
b = -2 * pt * bx
c = p0 ** 2 * (1 + bx ** 2) - pt ** 2

by0, by1 = numpy.roots([a, b, c])

p0_eq0 = (pt + bx * by0) / ((1 + bx ** 2) * (1 + by0 ** 2)) ** 0.5
p0_eq1 = (pt + bx * by1) / ((1 + bx ** 2) * (1 + by1 ** 2)) ** 0.5

print(f'{p0_eq0}')
print(f'{p0_eq1}')

alpha_y = by0 * Y_true.std() / W.std()

X_obs = X_true + alpha_x * W
Y_obs = Y_true + alpha_y * W

rho_true = pearsonr(X_true, Y_true)[0]
rho_obs = pearsonr(X_obs, Y_obs)[0]

print(f'{rho_true = }')
print(f'{rho_obs = }')

rho_XZ = -0.5  # 0.75
rho_YZ = 0.5  # 0.1
# rho_YZ_givenX = 0

print(f'{rho_XZ = }')
print(f'{rho_YZ = }')

Z = correlated_vector(arr=[X_obs, Y_obs], rho=[rho_XZ, rho_YZ], check=True)

# print(pearsonr(X_obs, Y_obs)[0])
# print(pearsonr(Z, X_obs)[0])
# print(pearsonr(Z, Y_obs)[0])
# print(pearsonr(X_obs, Y_obs)[0] * pearsonr(Z, X_obs)[0])
