from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, Div, Panel, Tabs, ColumnDataSource, RangeTool, DatetimeTickFormatter, LinearColorMapper, ColorBar, BasicTicker, Slider
from bokeh.transform import transform

import pathlib, sys

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class view_gapfilling:
    def __init__(self):
        print('Entrou view_bokeh_gapfilling02.py')

        self.div01 = Div(text=r'C:\Users\User\Desktop\teste02\dfs_compare.csv')

        self.path = TextInput(value='', title='Insert dfs path')

        self.button_view = Button(label='View')
        self.button_view.on_click(self._button_view)

        self.source_01 = ColumnDataSource(data=dict(date=[], time=[], ET=[]))

        self.fig_01 = figure(title='ET without Corrections', plot_height=350, plot_width=1200,
                             x_axis_type='datetime', y_axis_type='datetime')
        self.fig_01.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_01.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])
        #
        colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
        self.color_mapper = LinearColorMapper(palette=colors)
        #
        self.et_01 = self.fig_01.rect(x='date', y='time', fill_color=transform('ET', self.color_mapper),
                                      source=self.source_01, width=1000*60*60*24, height=1000*60*30, line_color=None)
        color_bar = ColorBar(color_mapper=self.color_mapper, ticker=BasicTicker(desired_num_ticks=len(colors)),label_standoff=6, border_line_color=None, location=(0,0))
        self.fig_01.add_layout(color_bar, 'right')

        self.button_gap = Button(label='Fill Gaps')
        self.button_gap.on_click(self._button_RFR)

        self.source_02 = ColumnDataSource(data=dict(date=[], time=[], ET=[]))
        self.fig_02 = figure(title='ET GapFilled', plot_width=1200, plot_height=350,
                             x_axis_type='datetime', y_axis_type='datetime', x_range=self.fig_01.x_range, y_range=self.fig_01.y_range)
        self.fig_02.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_02.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])

        self.et_02 = self.fig_02.rect(x='date', y='time', fill_color=transform('ET', self.color_mapper),
                                      source=self.source_02, width=1000*60*60*24, height=1000*60*30, line_color=None)
        # self.et_02_raw = self.fig_02.rect(x='date',y='')
        self.fig_02.add_layout(color_bar, 'right')



        curdoc().add_root(column(self.div01,
                                 self.path,
                                 self.button_view,
                                 self.fig_01,
                                 self.button_gap,
                                 self.fig_02))

    def _button_view(self):
        path = pathlib.Path('{}'.format(self.path.value))
        self.df = pd.read_csv(path, parse_dates=['TIMESTAMP', 'date_ns', 'time_ns'])

        self.df_na = self.df.loc[self.df['ET'].isna()].copy()

        self.df.dropna(inplace=True)

        self.source_01.data = dict(date=self.df['date_ns'],
                                   time=self.df['time_ns'],
                                   ET=self.df['ET'])

    def _button_RFR(self):
        column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e']

        X = self.df[column_x]
        y = self.df['ET']

        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

        et_model_RFR = RandomForestRegressor(random_state=1)
        et_model_RFR.fit(train_X, train_y)
        val_prediction_RFR = et_model_RFR.predict(val_X)
        mae_RFR = mean_absolute_error(val_y, val_prediction_RFR)
        print('MAE:', mae_RFR)

        self.df_na.dropna(subset=column_x, inplace=True)

        predict_gap = et_model_RFR.predict(self.df_na[column_x])
        self.df_na['ET_RFR'] = predict_gap

        self.source_02.data = dict(date=self.df_na['date_ns'],
                                   time=self.df_na['time_ns'],
                                   ET=self.df_na['ET_RFR'])


view_gapfilling()
