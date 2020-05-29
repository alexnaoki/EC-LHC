import ipywidgets
import numpy as np
import pandas as pd
import pathlib
from scipy.stats import linregress
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, RangeTool, Circle, Slope, Label, Legend, LegendItem, LinearColorMapper,SingleIntervalTicker, LinearAxis
from bokeh.layouts import gridplot, column, row
from bokeh.transform import transform

class view_et:
    def __init__(self):
        self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
 'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',
  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var','H_strg','LE_strg']
        TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


        file = r'C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01\eddypro_p00_fase01_full_output_2020-05-02T040616_adv.csv'
        df = pd.read_csv(file, skiprows=[0,2], na_values=-9999, usecols=self.ep_columns_filtered)
        # p

        # df = pd.DataFrame({'date':['2018','2018','2018'], 'time':['00:30','01:00','01:30'],'ET':[5,10,20]})


        print(df['ET'].min(), df['ET'].max(), df['ET'].mean())
        print(df['ET'].describe())
        # df['ET'] = df.loc[df['ET']<0]

        df2 = df.loc[(df['ET']>0)&(df['ET']<1), ['date','time','ET']]

        output_notebook()
        self.source_ep = ColumnDataSource(df2)
        colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']

        self.mapper = LinearColorMapper(palette=colors, low=df2['ET'].min(), high=df2['ET'].max())
        self.fig_01 = figure(title='testes',plot_width=1000, plot_height=600, x_range=df['date'].unique(),y_range=df['time'].unique(),
                         tools=TOOLS, tooltips=[('date','@date @time'),('ET','@ET')])
        self.fig_01.rect(source=self.source_ep, x='date',y='time',width=1,height=1, fill_color=transform('ET',self.mapper), line_color=None)
        # self.fig_01.xaxis.ticker.desired_num_ticks=10

        # xticker = SingleIntervalTicker(interval=10, num_minor_ticks=20)
        # xaxis = LinearAxis(ticker=xticker)
        # self.fig_01.add_layout(xaxis, 'below')


        self.fig_01.axis.axis_line_color = None
        self.fig_01.axis.major_tick_line_color = None

        self.fig_01.xaxis.major_label_orientation = 1
        # display()
        show(self.fig_01, notebook_handle=True)
