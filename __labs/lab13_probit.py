#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 28/07/2021

"""
import numpy
from matplotlib import pyplot
from scipy.stats import norm


# def logit(p):
#     if p == 1:
#         return 1
#     elif p == 0:
#         return 0
#     return numpy.log(p / (1 - p))


# def probit(p):
#     # return norm.ppf(p)
#     return Probit


def logit(x):
    return 1 / (1 + numpy.exp(-10 * x))


def is_treated(pr_treat, n=10_000):
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

n = 10000

# p = numpy.linspace(0.01, 0.99, n)

RV = numpy.linspace(-1, 1, n)  # running variable

y = list(numpy.zeros(int(n / 2))) + list(numpy.ones(int(n / 2)))

# fig, axes = pyplot.subplots(nrows=2)
# fig, ax = pyplot.subplots()

# sharp_rdd = numpy.apply_along_axis(sharp, axis=0, arr=X)
# fuzzy_rdd = numpy.apply_along_axis(fuzzy, axis=0, arr=X)
# sharp_rdd = numpy.vectorize(sharp)
# fuzzy_rdd = numpy.vectorize(fuzzy)

logit_vec = numpy.vectorize(logit)

prob_T = logit_vec(RV)

Y = numpy.random.randn(n)

Y /= numpy.linalg.norm(Y)

Y += delta * is_treated(prob_T)

fig, ax = pyplot.subplots()

ax.plot(RV, Y, marker="o", linestyle="")

pyplot.show()


# ax.plot(X, fuzzy_rdd(X), label='fuzzy')


# pr = Probit(Y, add_constant(X)).fit()

# ax.legend()

# axes[1].plot(p, probit(p))

# pyplot.show()
