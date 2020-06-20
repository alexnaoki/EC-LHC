from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, Div, Panel, Tabs, ColumnDataSource, RangeTool, DatetimeTickFormatter, LinearColorMapper, ColorBar, BasicTicker, Slider
from bokeh.transform import transform

import pathlib, sys

import pandas as pd
import numpy as np
import datetime as dt

class view_gapfilling:
    def __init__(self):
        print('Entrou view_bokeh_gapfilling.py')

        self.div01 = Div(text=r'C:\Users\User\Desktop\testse\df_filter.csv')

        self.path = TextInput(value='', title='Insert df Path:')

        self.button_view = Button(label='View')
        self.button_view.on_click(self._button_view)

        self.source_01 = ColumnDataSource(data=dict(date=[], time=[], ET=[]))

        self.fig_01 = figure(title='ET without Corrections', plot_height=350, plot_width=1200,
                             x_axis_type='datetime', y_axis_type='datetime')
        self.fig_01.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_01.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])

        colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
        self.color_mapper = LinearColorMapper(palette=colors)

        self.et_01 = self.fig_01.rect(x='date', y='time', fill_color=transform('ET', self.color_mapper),
                                      source=self.source_01, width=1000*60*60*24, height=1000*60*30, line_color=None)
        color_bar = ColorBar(color_mapper=self.color_mapper, ticker=BasicTicker(desired_num_ticks=len(colors)),label_standoff=6, border_line_color=None, location=(0,0))
        self.fig_01.add_layout(color_bar, 'right')

        self.slider_01 = Slider(start=0, end=30, value=1, step=1, title='# Adjecent Days')
        self.slider_01.on_change('value_throttled', lambda attr, old, new: self._mean_diurnal_gapFilling())



        self.source_02 = ColumnDataSource(data=dict(date=[], time=[], ET_mean=[]))
        self.fig_02 = figure(title='ET GapFilled', plot_width=1200, plot_height=350,
                             x_axis_type='datetime', y_axis_type='datetime')
        self.fig_02.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_02.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])

        self.et_02 = self.fig_02.rect(x='date', y='time', fill_color=transform('ET_mean', self.color_mapper),
                                      source=self.source_02, width=1000*60*60*24, height=1000*60*30, line_color=None)
        self.fig_02.add_layout(color_bar, 'right')


        curdoc().add_root(column(self.div01,
                                 self.path,
                                 self.button_view,
                                 self.fig_01,
                                 self.slider_01,
                                 self.fig_02))

    def _button_view(self):
        path = pathlib.Path('{}'.format(self.path.value))
        self.df = pd.read_csv(path, parse_dates=['TIMESTAMP','date_ns','time_ns'])

        print(self.df.columns.to_list())
        self.source_01.data = dict(date=self.df['date_ns'],
                                   time=self.df['time_ns'],
                                   ET=self.df['ET'])

        min_datetime = self.df['TIMESTAMP'].min()
        max_datetime = self.df['TIMESTAMP'].max()

        # self.df['date02'] = pd.to_datetime(self.df['TIMESTAMP'].dt.date)
        # self.df['time02'] = pd.to_datetime(self.df['TIMESTAMP'].dt.time, format="%H:%M:%S")
        # self.df['time02'] = pd.to_datetime

        df_full_timestamp = pd.DataFrame({"TIMESTAMP": pd.date_range(start=min_datetime, end=max_datetime, freq='30min')})
        self.df_merge = pd.merge(left=self.df, right=df_full_timestamp, on='TIMESTAMP', how='outer')

        self.df_merge['date02'] = pd.to_datetime(self.df_merge['TIMESTAMP'].dt.date)
        self.df_merge['time02'] = pd.to_datetime(self.df_merge['TIMESTAMP'].dt.time, format="%H:%M:%S")

    def _mean_diurnal_gapFilling(self):
        n_days = self.slider_01.value
        delta_days = [i for i in range(-n_days, n_days+1, 1)]

        self.df_na = self.df_merge[self.df_merge['ET'].isna()].copy()

        self.df_na['timestamp_adj'] = self.df_na['TIMESTAMP'].apply(lambda x: [x + dt.timedelta(days=i) for i in delta_days])

        for i,row in self.df_na.iterrows():
            self.df_na.loc[i, 'ET_mean'] = self.df_merge.loc[self.df_merge['TIMESTAMP'].isin(row['timestamp_adj']), 'ET'].mean()

        self.source_02.data = dict(date=self.df_na['date02'],
                                   time=self.df_na['time02'],
                                   ET_mean=self.df_na['ET_mean'])
        # print(self.df_na['ET_mean'])
view_gapfilling()
