#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 14/07/2021

"""

import numpy
import pandas
from matplotlib import pyplot, gridspec
from matplotlib.widgets import Slider, TextBox
from scipy.linalg import norm, svd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.tools import add_constant

pandas.set_option('display.max_columns', 100)

numpy.random.seed(123)


def scale_vector(arr):
    rmse = mean_squared_error(arr, [arr.mean()] * len(arr), squared=False)

    return arr / rmse


def get_pearsonr(x, y):
    return pearsonr(x, y)[0]


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

    sigma2 = sigma2.squeeze()

    if sigma2 < 0:
        raise ValueError(f'sigma2 = {sigma2.round(2)}: impossible correlation')

    X = dual_basis @ rho + numpy.sqrt(sigma2) * resid.reshape(-1, 1)

    X = (X / norm(X)).squeeze()

    if check:
        print(f'{sigma2 = }')
        print('Sanity check for identity matrix:\n', dual_basis.T @ scaled_vecs / n)

        # TODO: pretty table print with target rho and actual rho
        corrcoefs = numpy.apply_along_axis(get_pearsonr, axis=0, arr=scaled_vecs, y=X)

        for c in corrcoefs:
            print(c)

    return X


class IVPlot:

    def __init__(self):
        self.n = 1_000

        self.W = numpy.random.randn(self.n)  # confounder
        self.Z = numpy.random.randn(self.n)  # instrument

        self.W /= norm(self.W)
        self.Z /= norm(self.Z)

        self.make_figure()

        self.make_sliders()

        self.make_plot()

    def make_figure(self):
        gs = gridspec.GridSpec(2, 4)

        self.ax1 = pyplot.subplot(gs[0, 0])
        self.ax2 = pyplot.subplot(gs[0, 1])
        self.ax3 = pyplot.subplot(gs[0, 2])
        self.ax4 = pyplot.subplot(gs[0, 3])
        self.ax5 = pyplot.subplot(gs[1, 0:])

        self.ax1.set_xlabel('T')
        self.ax2.set_xlabel('Y_obs')
        self.ax3.set_xlabel('T')
        self.ax4.set_xlabel('Y_obs')
        self.ax5.set_xlabel('T')

        self.ax1.set_ylabel('W')
        self.ax2.set_ylabel('W')
        self.ax3.set_ylabel('Z')
        self.ax4.set_ylabel('Z')
        self.ax5.set_ylabel('Y')

        self.ax1.get_xaxis().set_ticks([])
        self.ax2.get_xaxis().set_ticks([])
        self.ax3.get_xaxis().set_ticks([])
        self.ax4.get_xaxis().set_ticks([])
        self.ax5.get_xaxis().set_ticks([])

        self.ax1.get_yaxis().set_ticks([])
        self.ax2.get_yaxis().set_ticks([])
        self.ax3.get_yaxis().set_ticks([])
        self.ax4.get_yaxis().set_ticks([])
        self.ax5.get_yaxis().set_ticks([])

    def make_sliders(self):
        rho_TW = 0.3
        rho_TZ = 0.2
        rho_YW = -0.5
        rho_YZ = 0.0
        rho_TY = 0.5

        axcolor = 'lightgoldenrodyellow'

        # ---------- ---------- ---------- ---------- ---------- ---------- TW

        ax1pos = self.ax1.get_position().corners()
        TW_pos = [
            ax1pos[0][0],  # x start
            ax1pos[1][1] + 0.01,  # y start
            ax1pos[2][0] - ax1pos[0][0],  # x length
            0.02,  # y length
        ]

        TW_ax = pyplot.axes(TW_pos, facecolor=axcolor)
        self.TW_slider = Slider(TW_ax, '', -1, 1, valinit=rho_TW, valstep=0.05)

        # ---------- ---------- ---------- ---------- ---------- ---------- YW

        ax2pos = self.ax2.get_position().corners()
        YW_pos = [
            ax2pos[0][0],  # x start
            ax2pos[1][1] + 0.01,  # y start
            ax2pos[2][0] - ax2pos[0][0],  # x length
            0.02,  # y length
        ]

        YW_ax = pyplot.axes(YW_pos, facecolor=axcolor)
        self.YW_slider = Slider(YW_ax, '', -1, 1, valinit=rho_YW, valstep=0.05)

        # ---------- ---------- ---------- ---------- ---------- ---------- TZ

        ax3pos = self.ax3.get_position().corners()
        TZ_pos = [
            ax3pos[0][0],  # x start
            ax3pos[1][1] + 0.01,  # y start
            ax3pos[2][0] - ax3pos[0][0],  # x length
            0.02,  # y length
        ]
        TZ_ax = pyplot.axes(TZ_pos, facecolor=axcolor)
        self.TZ_slider = Slider(TZ_ax, '', -1, 1, valinit=rho_TZ, valstep=0.05)

        # ---------- ---------- ---------- ---------- ---------- ---------- YZ

        ax4pos = self.ax4.get_position().corners()
        YZ_pos = [
            ax4pos[0][0],  # x start
            ax4pos[1][1] + 0.01,  # y start
            ax4pos[2][0] - ax4pos[0][0],  # x length
            0.02,  # y length
        ]

        YZ_ax = pyplot.axes(YZ_pos, facecolor=axcolor)
        self.YZ_slider = Slider(YZ_ax, '', -1, 1, valinit=rho_YZ, valstep=0.05)

        # ---------- ---------- ---------- ---------- ---------- ---------- TY

        ax5pos = self.ax5.get_position().corners()
        TY_pos = [
            ax5pos[0][0],  # x start
            ax5pos[0][1] - 0.05,  # y start
            ax5pos[2][0] - ax5pos[0][0],  # x length
            0.02,  # y length
        ]

        TY_ax = pyplot.axes(TY_pos, facecolor=axcolor)
        self.TY_slider = Slider(TY_ax, '', -1, 1, valinit=rho_TY, valstep=0.05)

        self.TW_slider.on_changed(self.update_TW_plot)
        self.YW_slider.on_changed(self.update_YW_plot)
        self.TZ_slider.on_changed(self.update_TZ_plot)
        self.YZ_slider.on_changed(self.update_YZ_plot)

    def prepare_vectors(self, vecs='all'):

        self.T = correlated_vector([self.W, self.Z], [self.TW_slider.val, self.TZ_slider.val], check=False)

        self.Y_true = correlated_vector_simple(self.T, self.TY_slider.val, check=False)

        self.Y_obs = self.Y_true + 2 * self.YW_slider.val * self.W + 2 * self.YZ_slider.val * self.Z

        self.Y_obs /= norm(self.Y_obs)

        # if vecs == 'T':
        #     # treatment variable
        #     self.T = correlated_vector([self.W, self.Z], [self.TW_slider.val, self.TZ_slider.val], check=False)
        #
        # elif vecs == 'Y':
        #     # observable response variable
        #     # self.Y_obs = correlated_vector([self.W, self.Z], [self.YW_slider.val, self.YZ_slider.val], check=False)
        #     # self.Y_obs = correlated_vector([self.T, self.W, self.Z], [self.TY_slider.val, self.YW_slider.val, self.YZ_slider.val], check=True)
        #     self.Y_true = correlated_vector_simple(self.T, self.TY_slider.val, check=False)
        #     self.Y_obs = self.Y_true + 3 * self.YW_slider.val * self.W
        #     self.Y_obs /= norm(self.Y_obs)
        #     print(pearsonr(self.Y_obs, self.W)[0])
        #     # self.Y_obs = correlated_vector([self.T, self.Z], [self.TY_slider.val, self.YZ_slider.val], check=True)
        #
        # elif vecs == 'all':
        #     self.T = correlated_vector([self.W, self.Z], [self.TW_slider.val, self.TZ_slider.val], check=False)
        #     # self.Y_obs = correlated_vector([self.W, self.Z], [self.YW_slider.val, self.YZ_slider.val], check=False)
        #     # self.Y_obs = correlated_vector([self.T, self.W, self.Z], [self.TY_slider.val, self.YW_slider.val, self.YZ_slider.val], check=True)
        #     self.Y_true = correlated_vector_simple(self.T, self.TY_slider.val, check=False)
        #     self.Y_obs = self.Y_true + 3 * self.YW_slider.val * self.W
        #     self.Y_obs /= norm(self.Y_obs)
        #     print(pearsonr(self.Y_obs, self.W)[0])
        #     # self.Y_obs = correlated_vector([self.T, self.Z], [self.TY_slider.val, self.YZ_slider.val], check=True)
        #
        # else:
        #     assert False, 'wrong argument passed'

    def make_plot(self):

        self.prepare_vectors()

        self.scatter_TW, = self.ax1.plot(self.T, self.W, color='C0', alpha=0.25, marker='o', linestyle='')
        self.scatter_YW, = self.ax2.plot(self.Y_obs, self.W, color='C0', alpha=0.25, marker='o', linestyle='')
        self.scatter_TZ, = self.ax3.plot(self.T, self.Z, color='C0', alpha=0.25, marker='o', linestyle='')
        self.scatter_YZ, = self.ax4.plot(self.Y_obs, self.Z, color='C0', alpha=0.25, marker='o', linestyle='')

        ols_TW = OLS(self.W, add_constant(self.T)).fit()
        ols_YW = OLS(self.W, add_constant(self.Y_obs)).fit()
        ols_TZ = OLS(self.Z, add_constant(self.T)).fit()
        ols_YZ = OLS(self.Z, add_constant(self.Y_obs)).fit()

        self.ols_TW, = self.ax1.plot(self.T, ols_TW.fittedvalues, color='C1')
        self.ols_YW, = self.ax2.plot(self.Y_obs, ols_YW.fittedvalues, color='C1')
        self.ols_TZ, = self.ax3.plot(self.T, ols_TZ.fittedvalues, color='C1')
        self.ols_YZ, = self.ax4.plot(self.Y_obs, ols_YZ.fittedvalues, color='C1')

        ols_YobsT = OLS(self.Y_obs, add_constant(self.T)).fit()
        olsYtrueT = OLS(self.Y_true, add_constant(self.T)).fit()
        iv_reg = IV2SLS(endog=self.Y_obs, exog=self.T, instrument=self.Z).fit()

        self.scatter_TY, = self.ax5.plot(self.T, self.Y_obs, color='C0', alpha=0.25, marker='o', linestyle='')
        self.ols_YobsT, = self.ax5.plot(self.T, ols_YobsT.fittedvalues, color='C1', label='naive reg')
        self.iv_reg, = self.ax5.plot(self.T, iv_reg.fittedvalues, color='C2', label='iv reg', linestyle='dotted')
        self.ols_YtrueT, = self.ax5.plot(self.T, olsYtrueT.fittedvalues, color='C3', label='true reg', linestyle='dashdot')

        self.ax5.legend()

    def update_bottom_plot(self):

        ols_YobsT = OLS(self.Y_obs, add_constant(self.T)).fit()
        olsYtrueT = OLS(self.Y_true, add_constant(self.T)).fit()
        iv_reg = IV2SLS(endog=self.Y_obs, exog=self.T, instrument=self.Z).fit()

        self.scatter_TY.set_data(self.T, self.Y_obs)
        self.ols_YobsT.set_data(self.T, ols_YobsT.fittedvalues)
        self.ols_YtrueT.set_data(self.T, olsYtrueT.fittedvalues)
        self.iv_reg.set_data(self.T, iv_reg.fittedvalues)

        self.ax5.relim()
        self.ax5.autoscale_view(True)

    def update_TW_plot(self, val):
        # if abs(val) != 1:
        self.prepare_vectors(vecs='T')
        # else:
          #  self.T = val * self.W

        print(self.scatter_TW)

        ols_TW = OLS(self.W, add_constant(self.T)).fit()
        self.scatter_TW.set_data(self.T, self.W)
        self.ols_TW.set_data(self.T, ols_TW.fittedvalues)

        self.ax1.relim()
        self.ax1.autoscale_view(True)

        self.update_bottom_plot()

        # ols_TZ = OLS(self.Z, add_constant(self.T)).fit()
        # self.scatter_TZ.set_data(self.T, self.Z)
        # self.ols_TZ.set_data(self.T, ols_TZ.fittedvalues)

    def update_YW_plot(self, val):
        # TODO: not working!
        # if abs(val) != 1:
        self.prepare_vectors(vecs='Y')
        # else:
            # self.Y_obs = val * self.W

        ols_YW = OLS(self.W, add_constant(self.Y_obs)).fit()

        self.scatter_YW.set_data(self.Y_obs, self.W)

        self.ols_YW.set_data(self.Y_obs, ols_YW.fittedvalues)

        self.ax2.relim()
        self.ax2.autoscale_view(True)

        self.update_bottom_plot()

    def update_TZ_plot(self, val):
        # if abs(val) != 1:
        self.prepare_vectors(vecs='T')
        # else:
           # self.T = val * self.Z

        ols_TZ = OLS(self.Z, add_constant(self.T)).fit()

        self.scatter_TZ.set_data(self.T, self.Z)

        self.ols_TZ.set_data(self.T, ols_TZ.fittedvalues)

        self.ax3.relim()
        self.ax3.autoscale_view(True)

        self.update_bottom_plot()

    def update_YZ_plot(self, val):
        # if abs(val) != 1:
        self.prepare_vectors(vecs='Y')
        #else:
         #   self.Y_obs = val * self.Z

        ols_YZ = OLS(self.Z, add_constant(self.Y_obs)).fit()

        self.scatter_YZ.set_data(self.Y_obs, self.Z)

        self.ols_YZ.set_data(self.Y_obs, ols_YZ.fittedvalues)

        self.ax4.relim()
        self.ax4.autoscale_view(True)

        self.update_bottom_plot()

    def print_corrs(self):

        corr_TW = round(pearsonr(self.T, self.W)[0], 2)
        corr_TZ = round(pearsonr(self.T, self.Z)[0], 2)
        corr_TY = round(pearsonr(self.T, self.Y_obs)[0], 2)
        corr_YW = round(pearsonr(self.Y_obs, self.W)[0], 2)
        corr_YZ = round(pearsonr(self.Y_obs, self.Z)[0], 2)

        print(f'{corr_TW = }')
        print(f'{corr_TZ = }\n')
        print(f'{corr_TY = }')
        print(f'{corr_YW = }')
        print(f'{corr_YZ = }')

if __name__ == '__main__':
    ivplot = IVPlot()
    mng = pyplot.get_current_fig_manager()
    mng.window.showMaximized()
    px = mng.canvas.width()
    mng.window.move(px, 0)

# TW_slider.on_changed(update_plot)
# YW_slider.on_changed(update_plot)
# TZ_slider.on_changed(update_plot)
# YZ_slider.on_changed(update_plot)
# TY_slider.on_changed(update_plot)
#
# pyplot.show()
