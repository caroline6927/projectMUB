# heatmap
# AC plot
# matplotlib plots

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot
from plotly import tools
import plotly.plotly as plty
import plotly.graph_objs as go
from bokeh.plotting import figure, output_file

tools.set_credentials_file(username='caroline6927', api_key='bdAo5ZQ6WvRR2hMg7h0s')

# for jupyter notebook
# import pylab as plt
# %pylab inline
# pylab.rcParams['figure.figsize'] = (20, 7)


def get_heatmap(data, var, title):
    x = [t.date() for t in data.date_time]
    x = list(set(x))
    x = pd.Series(x).sort_values().tolist()
    y = pd.date_range('00:00:00', freq='15min', periods=96)
    z = []
    for t in y:
        new_row = []
        for d in x:
            key = pd.to_datetime(str(d) + ' ' + str(t.time()))
            try:
                z_data = data.loc[data['date_time'] == key, var]
                if len(z_data) > 0:
                    new_row.append(list(z_data)[0])
                else:
                    new_row.append(np.nan)
            except KeyError:
                new_row.append(np.nan)
        z.append(list(new_row))
    map_data = [
        go.Heatmap(
            z=z,
            x=x,
            y=list(range(0, 96)),
            colorscale='Viridis',
        )]

    layout = go.Layout(
        title=var,
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks='')
    )

    fig = go.Figure(data=map_data, layout=layout)
    plty.iplot(fig, filename=title)


def get_plot_acf(data, lags, interval, title='acf', mode='acf'):
    if mode == 'acf':
        plot_acf(data, lags=lags, alpha=.05)
    if mode == 'pacf':
        plot_pacf(data, lags=lags, alpha=.05)
    pyplot.xticks(np.arange(0, lags, interval))
    pyplot.title(title)
    pyplot.show()


def plot_data(data, option='plain'):
    if option == 'plain':
        pyplot.plot(data, '.')
        pyplot.show()
    if option == 'rich':
        output_file("plot.html")
        p = figure(plot_width=1000, plot_height=400)
        p.line(list(range(len(data))), data, line_width=2, color="navy", alpha=0.5)
        show(p)
