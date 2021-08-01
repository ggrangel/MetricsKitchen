#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 28/07/2021

"""
import os

import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

import matplotlib.pyplot as plt

# def logit(p):
#     if p == 1:
#         return 1
#     elif p == 0:
#         return 0
#     return numpy.log(p / (1 - p))


# def probit(p):
#     # return norm.ppf(p)
#     return Probit


def logit(x, alpha=1000):
    return 1 / (1 + numpy.exp(-alpha * x))


def is_treated(pr_treat, n=1_000):
    u = numpy.random.uniform(size=n)

    return pr_treat > u


# def sharp(x, C0=0):
#     if x > C0:
#         return 1
#     return 0
#
#
# def fuzzy(x, C0=0):
#     # delta = abs(x - C0)
#     x = (x + 1) / 2
#     return probit(x) + 0.5
#

delta = 0.3

n = 1000

frame = pandas.DataFrame(columns=["Y", "RV", "is_treated", "error"])

frame["RV"] = numpy.linspace(-1, 1, n)  # running variable

logit_vec = numpy.vectorize(logit)

frame["is_treated"] = logit_vec(frame.RV)

frame["error"] = numpy.random.randn(n)

frame["Y"] = frame.RV + frame.is_treated + 0.5 * frame.error

fig, ax = pyplot.subplots()

ax.plot(frame.RV, frame.Y, marker="o", linestyle="", alpha=0.1)

before = frame[frame.RV < 0]
after = frame[frame.RV > 0]

ols_before = OLS(before.Y, add_constant(before.RV)).fit()
ols_after = OLS(after.Y, add_constant(after.RV)).fit()

last_point_before = ols_before.fittedvalues.values[-1]
first_point_after = ols_after.fittedvalues.values[0]

delta = first_point_after - last_point_before

print(delta)

ax.plot(before.RV, ols_before.fittedvalues, color="C1")
ax.plot(after.RV, ols_after.fittedvalues, color="C1")

x1, y1 = after.RV.values[0], first_point_after
x2, y2 = before.RV.values[-1], last_point_before

# ax.plot([x1, x2], [y1, y2], ".")
connectionstyle = "bar, fraction=-0.3"
arrow = ax.annotate(
    "",
    xy=(x1, y1),
    xycoords="data",
    xytext=(x2, y2),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="->",
        color="C3",
        shrinkA=5,
        shrinkB=5,
        patchA=None,
        patchB=None,
        connectionstyle=connectionstyle,
        linestyle="dashed",
    ),
)

ax.annotate(
    rf"$\Delta=${round(delta, 2)}",
    xy=(arrow.get_position()[0] - 0.3, (y1 + y2) / 2),
    color="C3",
    # xytext=arrow.get_position(),
    # textcoords="data",
)


pyplot.show()
# ax.plot(X, fuzzy_rdd(X), label='fuzzy')


# pr = Probit(Y, add_constant(X)).fit()

# ax.legend()

# axes[1].plot(p, probit(p))

# pyplot.show()
