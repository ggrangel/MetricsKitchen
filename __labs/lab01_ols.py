#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 01/07/2021

"""

import numpy
import pandas
from matplotlib import pyplot
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from statsmodels.regression.linear_model import OLS

# ---------- ---------- ---------- ---------- ---------- ---------- constants

numpy.random.seed(123)

n_data = 1000

t_max = 100

# ---------- ---------- ---------- ---------- ---------- ---------- initial conditions

s0 = 50  # sample size
e0 = 50  # error factor
error0 = numpy.random.randn(n_data)

# ========== ========== ========== ========== ========== ========== data


def data_generation_process(t):
    a, b = 0, 1

    return t / t_max + numpy.sin(a + b * t - 0.007 * t ** 2)


def get_plot_data(error):
    x = numpy.linspace(0, t_max, n_data)

    frame = pandas.DataFrame({
        'X0': 1,
        'X1': x,
        'Y': data_generation_process(x),
        'error': error,
    })

    frame['Y+error'] = frame.Y + e0 * frame.error / 100

    return frame


# ---------- ---------- ---------- ---------- ---------- ---------- regressions

DATA = get_plot_data(error0)

true_ols = OLS(DATA.Y, DATA.loc[:, ['X0', 'X1']]).fit()

data_sample = DATA.sample(frac=s0 / 100)

ols_sample = OLS(data_sample['Y+error'], data_sample.loc[:, ['X0', 'X1']]).fit()

# ========== ========== ========== ========== ========== ========== figure

fig, ax = pyplot.subplots()

fig.set_size_inches([8.89, 8.24])

pyplot.subplots_adjust(left=0.25, bottom=0.25)

# ---------- ---------- ---------- ---------- ---------- ---------- axes

dgp_ax, = ax.plot(DATA.X1, DATA.Y, label='DGP', color='C0')

pop_ax, = ax.plot(DATA.X1, DATA['Y+error'], linestyle='', marker='o', alpha=0.2, label='Population', color='C1')
pop_ax.set_visible(False)

true_reg_ax, = ax.plot(DATA.X1, true_ols.fittedvalues, linestyle='dashed', label='True Regresison', color='C2')

sample_ax, = ax.plot(data_sample.X1, data_sample['Y+error'], marker='o', linestyle='', alpha=0.5, label='Sample',
                     color='C4')

est_reg_ax, = ax.plot(data_sample.X1, ols_sample.fittedvalues, linestyle='dashed', label='Est. Regresison',
                      color='C3')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)

ax.set_title('OLS: true and estimated regression')


# ========== ========== ========== ========== ========== ========== widgets

def update_plot(val):
    sample_size = sample_size_slider.val / 100

    error_scale = error_scale_slider.val / 100

    DATA['Y+error'] = DATA.Y + error_scale * DATA.error

    data_sample_ = DATA.sample(frac=sample_size)

    if len(data_sample_) < 2:
        data_sample_ = DATA.sample(n=2)
        sample_size_slider.val = 2 / len(DATA)

    ols_sample_ = OLS(data_sample_['Y+error'], data_sample_.loc[:, ['X0', 'X1']]).fit()

    est_reg_ax.set_data(data_sample_.X1, ols_sample_.fittedvalues)

    sample_ax.set_data(data_sample_.X1, data_sample_['Y+error'])

    fig.canvas.draw_idle()


# ---------- ---------- ---------- ---------- ---------- ---------- sliders

axcolor = 'lightgoldenrodyellow'

sample_size_ax = pyplot.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
error_scale_ax = pyplot.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)

sample_size_slider = Slider(sample_size_ax, 'Sample Size (%)', 0, 100, valinit=s0)
error_scale_slider = Slider(error_scale_ax, 'Measurement Error (%)', 0, 100, valinit=e0)

sample_size_slider.on_changed(update_plot)
error_scale_slider.on_changed(update_plot)


# ---------- ---------- ---------- ---------- ---------- ---------- check buttons

def check_bttns_func(label):
    index = labels.index(label)

    axes[index].set_visible(not axes[index].get_visible())

    pyplot.draw()

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)


axes = [dgp_ax, true_reg_ax, est_reg_ax, pop_ax, sample_ax]

cb_ax = pyplot.axes([0.02, 0.65, 0.18, 0.15])

labels = [str(ax.get_label()) for ax in axes]

visibility = [ax.get_visible() for ax in axes]

checkbttn = CheckButtons(cb_ax, labels, visibility)

checkbttn.on_clicked(check_bttns_func)


# ---------- ---------- ---------- ---------- ---------- ---------- radio buttons

def radio_bttns_func(label):
    DATA['error'] = RB_DICT[label]

    update_plot(None)


rb_ax = pyplot.axes([0.02, 0.45, 0.18, 0.1])

r1 = r'$\varepsilon \sim \mathcal{N}(0, 1)$'
r2 = r'$\varepsilon \sim \mathcal{N}(0, t/500)$'

RB_DICT = {
    r1: numpy.random.randn(n_data),
    r2: numpy.random.normal(loc=0, scale=numpy.sqrt(numpy.arange(n_data) / 500), size=n_data),
}

radio = RadioButtons(rb_ax, (r1, r2))

radio.on_clicked(radio_bttns_func)

# ---------- ---------- ---------- ---------- ---------- ---------- resample button

resample_ax = pyplot.axes([0.85, 0.01, 0.12, 0.03])

resample_bttn = Button(resample_ax, 'Resample', color=axcolor, hovercolor='0.975')

resample_bttn.on_clicked(update_plot)
