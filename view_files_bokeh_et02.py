from bokeh.io import show, output_file, curdoc
from bokeh.models import Button, TextInput, Paragraph, Select, Panel, Tabs,ColumnDataSource, RangeTool, Circle, Slope, Label, Legend, LegendItem, LinearColorMapper, Div,CheckboxButtonGroup,Slider,CheckboxGroup,RangeSlider
from bokeh.layouts import gridplot, column, row
from bokeh.plotting import figure
from bokeh.transform import transform


import pathlib
import pandas as pd
import numpy as np
import datetime as dt

class view_et:
    def __init__(self):

        self.ep_columns_filtered = ['date','time','ET']
        #
        #
        file = r'C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01\eddypro_p00_fase01_full_output_2020-05-02T040616_adv.csv'
        self.df = pd.read_csv(file, skiprows=[0,2], na_values=-9999, usecols=self.ep_columns_filtered)

        button = Button(label='Plot')
        button.on_click(self._button)

        self.source = ColumnDataSource(data=dict(date=[], time=[], et=[]))
        # self.source = ColumnDataSource(self.df)

        self.fig = figure(title='ET', plot_height=500, plot_width=500, x_range=[(dt.datetime(2018,1,1) + dt.timedelta(days=i)).date().strftime('%Y-%m-%d') for i in range(1,720)], y_range=[(dt.datetime(2000,1,1) + dt.timedelta(minutes=i*30)).time().strftime('%H:%M') for i in range(48)])
        colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
        self.mapper = LinearColorMapper(palette=colors, low=self.df.ET.min(), high=self.df.ET.max())

        self.fig.rect(x='date', y='time', fill_color=transform('et', self.mapper),width=1, height=1, source=self.source)


        curdoc().add_root(column(button, self.fig))

    def _button(self):
        print('foi')

    #     self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
    # 'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',
    # 'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var','H_strg','LE_strg']
    #     TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    #
    # #
    #     file = r'C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01\eddypro_p00_fase01_full_output_2020-05-02T040616_adv.csv'
    #     self.df = pd.read_csv(file, skiprows=[0,2], na_values=-9999, usecols=self.ep_columns_filtered)
        # self.fig.x_range = self.df['date'].unique()
        # self.fig.y_range = self.df['time'].unique()
        self.source.data = dict(date=self.df['date'],
                                time=self.df['time'],
                                et=self.df['ET'])
        # print(self.source.data)
        print('foi2')


view_et()
