#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 01/07/2021

"""
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot, gridspec
from scipy.linalg import norm, svd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.sandbox.regression.gmm import IV2SLS

pandas.set_option('display.max_columns', 100)

numpy.random.seed(123)

n = 1000

a, b, sigma_error = 0, 1, 20

dgp = lambda _t: (a + b * _t + 15 * numpy.sin(0.326 * _t) - 0.007 * _t ** 2)  # dgp = data generation process

t1 = numpy.linspace(0, 80, n)
t2 = numpy.linspace(0, 20, n)
# w = numpy.linspace(0, 50, n)
w = 10 + numpy.random.randn(n)*50
data = pandas.DataFrame({
    'X0': 1,
    't1': t1,
    't2': t2,
    'W': w,
    'Y': dgp(t1 + t2),
    'error': sigma_error * numpy.random.randn(n),
})

data['X1'] = t1 + t2

data['population'] = data.Y + data.error

ols_pop = OLS(data.population, data.loc[:, ['X0', 'X1']]).fit()

data['estimated'] = ols_pop.params[0] + ols_pop.params[1] * data.X1

data['residual'] = data['Y'] - data['estimated']

fig, ax = pyplot.subplots()

ax.plot(data.X1, data.Y, label='True DGP', color='C0')

ax.plot(data.X1, ols_pop.fittedvalues, linestyle='dashed', label='True Regresison', color='C2')

data.X1 += data.W
data.Y -= data.W
data['population'] = data.Y + data.error

data_sample = data.query('0 <= X1 <= 100')

sample_size = 0.1

data_sample = data_sample.sample(frac=sample_size)

ols_sample = OLS(data_sample.population, data_sample.loc[:, ['X0', 'X1']]).fit()

data_sample['estimated'] = ols_sample.params[0] + ols_sample.params[1] * data_sample.X1

data_sample['residual'] = data_sample['Y'] - data_sample['estimated']

ax.plot(data_sample.X1, ols_sample.fittedvalues, linestyle='dashed', label='Estimated Regresison', color='C3')
# ax.plot(data_sample.X1, data_sample.population, marker='o', linestyle='', alpha=0.1, label='Population', color='C1')
ax.plot(data_sample.X1, data_sample.population, marker='o', linestyle='', alpha=0.5, label='Sample', color='C4')

ax.set_title('OLS: true vs estimated regression')

pyplot.show()

stage1 = OLS(data_sample.X1, data_sample[['X0', 'X1']]).fit()

stage2 = OLS(data_sample.Y_obs, stage1.fittedvalues).fit()

# est = data_sample[['X0', 'X1']] @ stage2.params.values.reshape(2, -1)
est = data_sample[['X1']] @ stage2.params.values.reshape(1, -1)

ax.plot(data_sample.X1.values, est.values, linestyle='dashed', label='IV Regresison', color='C5')

ax.legend()


