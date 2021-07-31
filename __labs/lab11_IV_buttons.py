#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 14/07/2021

"""
import sys
from math import isclose

import numpy
import pandas
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib import pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy.linalg import norm, svd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.tools import add_constant

pandas.set_option('display.max_columns', 100)


# numpy.random.seed(123)

class QCorrSlider(QSlider):

    def __init__(self):
        super().__init__(Qt.Horizontal)

        self.setRange(-100, 100)
        self.setSingleStep(5)
        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(25)


class MainView(QDialog):

    def __init__(self, rho_init):
        super().__init__()

        self.rho_init = rho_init

    def setup(self, controller):
        self.setWindowTitle('Instrumental Variables')

        # ========== ========== ========== ========== ========== ==========  make layouts

        self.lytMain = QHBoxLayout()

        self.lytLeft = QVBoxLayout()

        self.lytSpinBoxes = QFormLayout()

        self.lytSlider = QFormLayout()

        # ========== ========== ========== ========== ========== ==========  make widgets

        # ---------- ---------- ---------- ---------- ---------- ---------- DAG

        self.lblDAG = QLabel()

        pixmapDAG = QPixmap('dag.png')

        self.lblDAG.setPixmap(pixmapDAG)
        self.lblDAG.resize(pixmapDAG.width(), pixmapDAG.height())

        self.canvas = FigureCanvasQTAgg(pyplot.Figure())

        # ---------- ---------- ---------- ---------- ---------- ---------- simulate groupbox

        self.gbxSimulate = QGroupBox()

        self.gbxSimulate.setTitle("Treatment and response correlation coefficient")

        self.spinbTY_true = QDoubleSpinBox()
        self.spinbTY_obs = QDoubleSpinBox()

        self.spinbTY_true.setRange(-1, 1)
        self.spinbTY_true.setSingleStep(0.05)

        self.spinbTY_obs.setRange(-1, 1)
        self.spinbTY_obs.setSingleStep(0.05)

        self.btnSimulate = QPushButton('Simulate')

        # ---------- ---------- ---------- ---------- ---------- ---------- instrument corr. coef. groupbox

        self.gbxInstrument = QGroupBox()

        self.gbxInstrument.setTitle("Instrument correlation coefficient")

        self.sldTZ = QCorrSlider()
        self.sldYZ = QCorrSlider()

        # ========== ========== ========== ========== ========== ==========  setup layouts

        # ---------- ---------- ---------- ---------- ---------- ---------- main layout

        self.setLayout(self.lytMain)

        self.lytMain.addLayout(self.lytLeft)

        self.lytMain.addWidget(self.canvas)

        # ---------- ---------- ---------- ---------- ---------- ---------- left layout

        self.lytLeft.addWidget(self.lblDAG)

        self.lytLeft.setAlignment(self.lblDAG, Qt.AlignCenter)

        self.lytLeft.addStretch()

        self.lytLeft.addWidget(self.gbxSimulate)

        self.lytLeft.addStretch()

        self.lytLeft.addWidget(self.gbxInstrument)

        self.lytLeft.addStretch()

        # ---------- ---------- ---------- ---------- ---------- ---------- simulate layout

        self.gbxSimulate.setLayout(self.lytSpinBoxes)

        self.lytSpinBoxes.addRow("True ", self.spinbTY_true)
        self.lytSpinBoxes.addRow("Observed ", self.spinbTY_obs)

        self.lytSpinBoxes.addWidget(self.btnSimulate)

        self.lytSpinBoxes.setAlignment(self.btnSimulate, Qt.AlignRight)

        # ---------- ---------- ---------- ---------- ---------- ---------- instrument layout

        self.gbxInstrument.setLayout(self.lytSlider)

        self.lytSlider.addRow("(T, Z)", self.sldTZ)
        self.lytSlider.addRow("(Y, Z), given X", self.sldYZ)

        # ========== ========== ========== ========== ========== ========== initial values

        self.spinbTY_true.setValue(self.rho_init[0])
        self.spinbTY_obs.setValue(self.rho_init[1])

        self.sldTZ.setValue(int(self.rho_init[2] * 100))
        self.sldYZ.setValue(int(self.rho_init[3] * 100))

        # ========== ========== ========== ========== ========== ==========  events and slots

        self.btnSimulate.clicked.connect(controller.evt_btnSimulate_clicked)

        self.sldTZ.valueChanged.connect(controller.evt_sldTZ_valueChanged)
        self.sldYZ.valueChanged.connect(controller.evt_sldYZ_valueChanged)

    def make_plot(self, T_true, T_obs, Y_obs, Y_true, Z, **kwargs):
        self.ax = self.canvas.figure.subplots()

        # ---------- ---------- ---------- ---------- ---------- ---------- compute regressions

        true_ols = OLS(Y_true, add_constant(T_true)).fit()

        naive_ols = OLS(Y_obs, add_constant(T_obs)).fit()

        iv_reg = IV2SLS(endog=Y_obs, exog=add_constant(T_obs), instrument=add_constant(Z)).fit()

        # ---------- ---------- ---------- ---------- ---------- ---------- plot regressions

        self.plot_naive, = self.ax.plot(T_obs, naive_ols.fittedvalues, color='C1',
                                        label='naive reg')

        self.plot_true, = self.ax.plot(T_obs, true_ols.predict(add_constant(T_obs)), color='C2', label='true reg',
                                       linestyle='dashed')

        self.plot_iv, = self.ax.plot(T_obs, iv_reg.fittedvalues, color='C3', label='IV reg',
                                     linestyle='dotted')

        # ---------- ---------- ---------- ---------- ---------- ---------- configure axes

        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])

        self.ax.set_xlabel('T')
        self.ax.set_ylabel('Y')

        self.ax.legend(loc='upper left')

        self.ax.relim()
        self.canvas.draw_idle()

    def update_plot(self, T_obs, Y_obs, Z, **kwargs):
        iv_reg = IV2SLS(endog=Y_obs, exog=add_constant(T_obs), instrument=add_constant(Z)).fit()

        self.plot_iv.set_data(T_obs, iv_reg.fittedvalues)

        self.ax.relim()
        self.canvas.draw_idle()


class Model:

    def __init__(self, n=10_000):
        self.n = n

    def make_variables(self, rho_TY_true, rho_TY_obs, rho_TZ, rho_YZ):
        cov = [[1, rho_TY_true], [rho_TY_true, 1]]

        samples = numpy.random.multivariate_normal(mean=[0, 0], cov=cov, size=self.n).T

        T_true, Y_true = samples[0], samples[1]

        self.variables = {
            'T_true': T_true,
            'Y_true': Y_true,
        }

        self.make_confounded_var(rho_TY_true, rho_TY_obs)

        self.make_instrument(rho_TY_true, rho_TZ, rho_YZ)

    def make_confounded_var(self, rho_t, rho_o):
        """

        :param rho_t: true correlation coeff between T and Y
        :param rho_o: observed correlation coeff between T and Y
        :return:
        """
        W = numpy.random.randn(self.n)

        alpha_x = 1

        bx = alpha_x * W.std() / self.variables['T_true'].std()

        a = (rho_o ** 2 - 1) * bx ** 2 + rho_o ** 2
        b = -2 * rho_t * bx
        c = rho_o ** 2 * (1 + bx ** 2) - rho_t ** 2

        by1, by2 = numpy.roots([a, b, c])

        po1 = (rho_t + bx * by1) / ((1 + bx ** 2) * (1 + by1 ** 2)) ** 0.5
        po2 = (rho_t + bx * by2) / ((1 + bx ** 2) * (1 + by2 ** 2)) ** 0.5

        if isclose(po1.real, rho_o):
            by = by1.real
        else:  # isclose(po2, rho_o)
            by = by2.real

        alpha_y = by * self.variables['Y_true'].std() / W.std()

        self.variables['T_obs'] = self.variables['T_true'] + alpha_x * W
        self.variables['Y_obs'] = self.variables['Y_true'] + alpha_y * W

    def make_instrument(self, rho_TY, rho_TZ, partial_YZ):
        rho_YZ = get_corr_given_partial(rho_TY, rho_TZ, partial_YZ)

        Z = correlated_vector([self.variables['T_true'], self.variables['Y_true']], rho=[rho_TZ, rho_YZ], check=True)

        self.variables['Z'] = Z


class Controller:

    def __init__(self, model: Model, view: MainView):
        self.model = model
        self.view = view

        self.view.setup(self)

        self.evt_btnSimulate_clicked()

        self.view.show()

    def evt_btnSimulate_clicked(self):
        self.model.make_variables(
            self.view.spinbTY_true.value(),
            self.view.spinbTY_obs.value(),
            self.view.sldTZ.value() / 100,
            self.view.sldYZ.value() / 100
        )

        self.view.make_plot(**self.model.variables)

    def sliders_changed(self):
        self.model.make_instrument(
            self.view.spinbTY_true.value(),
            self.view.sldTZ.value() / 100,
            self.view.sldYZ.value() / 100
        )

        self.view.update_plot(**self.model.variables)

    def evt_sldTZ_valueChanged(self):
        self.sliders_changed()

    def evt_sldYZ_valueChanged(self):
        self.sliders_changed()


def get_partial_correlation(X, Y, Z):
    """Computes the partial correlation of X and Y, given Z"""
    olsX = OLS(X, add_constant(Z)).fit()
    olsY = OLS(Y, add_constant(Z)).fit()

    a = (get_pearsonr(X, Y) - get_pearsonr(X, Z) * get_pearsonr(Z, Y)) / (
            numpy.sqrt(1 - get_pearsonr(X, Z) ** 2) * numpy.sqrt(1 - get_pearsonr(Z, Y) ** 2))

    b = pearsonr(olsX.resid, olsY.resid)[0]

    return b


def get_corr_given_partial(rho_TY, rho_TZ, partial_YZ):
    c = partial_YZ * numpy.sqrt(1 - rho_TY ** 2) * numpy.sqrt(1 - rho_TZ ** 2) + rho_TY * rho_TZ

    return c


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

    # scaled_vecs = numpy.apply_along_axis(scale_vector, axis=1, arr=arr).T
    scaled_vecs = arr.T

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

    I = dual_basis.T @ scaled_vecs / n

    if check:
        # print(f'{sigma2 = }')
        print('Sanity check for identity matrix:\n', dual_basis.T @ scaled_vecs / n)

        # TODO: pretty table print with target rho and actual rho
        corrcoefs = numpy.apply_along_axis(get_pearsonr, axis=0, arr=scaled_vecs, y=X)

        # for c in corrcoefs:
        #     print(c)

    return X


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainView(rho_init=[-0.3, 0.45, 0.15, 0.2])

    # starts app in the left monitor
    monitor = QDesktopWidget().screenGeometry(1)
    window.move(monitor.left(), monitor.top())

    print(QStyleFactory.keys())

    c = Controller(Model(n=50_000), window)

    sys.exit(app.exec_())
