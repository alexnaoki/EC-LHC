from bokeh.io import show, output_file, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column, row
from bokeh.models import Slider, ColumnDataSource, Button, Tabs, Panel, DateSlider, Range1d, Div, TextInput, Select, Panel

import sys, pathlib
sys.path.append(str(pathlib.Path(sys.path[0]).parent/'footprint'/'FFP_python_v1_4'))

from calc_footprint_FFP_adjusted01 import FFP
import numpy as np
import pandas as pd
import datetime as dt

class view_k15:
    def __init__(self):
        self.a = FFP()

        self.tabs = Tabs(tabs=[self.tab_01()])

        self.source = ColumnDataSource(data=dict(xrs=[[1,2,3],[10,11,12]], yrs=[[1,2,3],[5,3,2]]))

        fig01 = figure(title='K15', plot_height=500, plot_width=500)
        mlines = fig01.multi_line(xs='xrs',ys='yrs', source=self.source)

        file = r'C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01\eddypro_p00_fase01_full_output_2020-05-02T040616_adv.csv'
        self.df = pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date','time']})
        self.adjust_wind_direction()

        self.source_02 = ColumnDataSource(data=dict(xrs=[], yrs=[]))

        self.datetime_slider = DateSlider(title='Datetime', start=self.df['TIMESTAMP'].min(), end=self.df['TIMESTAMP'].max(), value=self.df['TIMESTAMP'].min(), step=1000*60*30, format='%x %X')
        self.datetime_slider.on_change('value_throttled', lambda attr,old, new: self.update())

        self.div_inputs = Div(text='Sem dados', width=500)

        fig02 = figure(title='FullOutput', plot_height=500, plot_width=500)
        fig02.x_range = Range1d(-1000,1000)
        fig02.y_range = Range1d(-1000,1000)
        mlines02 = fig02.multi_line(xs='xrs', ys='yrs', source=self.source_02)

        curdoc().add_root(row(column(self.tabs,fig01), column(self.datetime_slider, self.div_inputs, fig02)))

    def tab_01(self):
        self.slider_zm = Slider(title='zm', start=0, end=20, step=0.1, value=1)

        self.slider_umean = Slider(title='Umean', start=0.1, end=10, step=0.01, value=1)

        self.slider_h = Slider(title='h', start=100, end=5000, step=1, value=1000)

        self.slider_ol = Slider(title='ol', start=-1000, end=1000, step=0.01, value=15)

        self.slider_sigmav = Slider(title='sigmav', start=0, end=5, step=0.1, value=0.5)

        self.slider_ustar = Slider(title='ustar', start=0.01, end=5, step=0.01, value=1)

        self.slider_wind_dir = Slider(title='wind dir', start=0, end=359, step=0.1,value=0)

        sliders = [self.slider_zm, self.slider_umean, self.slider_h,self.slider_ol,self.slider_sigmav, self.slider_ustar, self.slider_wind_dir]
        for slider in sliders:
            slider.on_change('value_throttled', lambda attr, old, new: self.run_footprint())

        tab01 = Panel(child=column(self.slider_zm,
                                   self.slider_umean,
                                   self.slider_h,
                                   self.slider_ol,
                                   self.slider_sigmav,
                                   self.slider_ustar,
                                   self.slider_wind_dir), title='Inputs K15')
        return tab01

    def run_footprint(self):
        out = self.a.output(zm=self.slider_zm.value,
                            umean=self.slider_umean.value,
                            h=self.slider_h.value,
                            ol=self.slider_ol.value,
                            sigmav=self.slider_sigmav.value,
                            ustar=self.slider_ustar.value,
                            wind_dir=self.slider_wind_dir.value,
                            rs=[0.3, 0.9], crop=False, fig=False)

        self.source.data = dict(xrs=out[8], yrs=out[9])

    def adjust_wind_direction(self):
        self.df['wind_dir_sonic'] = 360 - self.df['wind_dir']
        azimute = 135.1
        self.df['wind_dir_compass'] = (360 + azimute - self.df['wind_dir_sonic']).apply(lambda x: x-360 if x>=360 else x)

    def update(self):
        datetime = dt.datetime.utcfromtimestamp(self.datetime_slider.value/1000)

        inputs = self.df.loc[self.df['TIMESTAMP']==datetime, ['u_rot','L','u*','v_var','wind_dir_compass']]
        print(inputs)
        self.div_inputs.text = '''
        <table border="2"><tbody><tr>
			<td>&nbsp;zm</td>
			<td>umean</td>
			<td>h</td>
			<td>ol</td>
			<td>sigmav</td>
			<td>ustar</td>
			<td>wind_dir_compass</td>
		</tr><tr>
			<td>&nbsp;{}</td>
			<td>{}&nbsp;</td>
			<td>{}</td>
			<td>{}</td>
			<td>{}</td>
			<td>{}</td>
			<td>&nbsp;{}</td>
		</tr></tbody></table>
        '''.format(9,
                   inputs['u_rot'].values[0],
                   1000,
                   inputs['L'].values[0],
                   inputs['v_var'].values[0],
                   inputs['u*'].values[0],
                   inputs['wind_dir_compass'].values[0])

        out = self.a.output(zm=9,
                            umean=inputs['u_rot'].values[0],
                            h=1000,
                            ol=inputs['L'].values[0],
                            sigmav=inputs['v_var'].values[0],
                            ustar=inputs['u*'].values[0],
                            wind_dir=inputs['wind_dir_compass'].values[0],
                            rs=[0.3, 0.9], crop=False, fig=False)

        self.source_02.data = dict(xrs=out[8], yrs=out[9])


view_k15()
