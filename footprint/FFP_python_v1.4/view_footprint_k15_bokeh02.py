from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import Slider, ColumnDataSource, Button, Tabs, Panel, DateSlider, Range1d, Div, TextInput, Select, Panel, DateRangeSlider
from bokeh.layouts import column, row

from calc_footprint_FFP_adjusted01 import FFP
from calc_footprint_FFP_climatology_adjusted01 import FFP_climatology
import numpy as np
import pandas as pd
import pathlib
import datetime as dt

class view_k15:
    def __init__(self):
        self.ep_columns_filtered = ['date','time','wind_dir', 'u_rot','L','v_var','u*','wind_speed']

        self.tabs = Tabs(tabs=[self.tab_01(), self.tab_02(), self.tab_03()])


        curdoc().add_root(self.tabs)

    def tab_01(self):
        self.div_01 = Div(text=r'C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01', width=500)

        self.path_ep = TextInput(value='', title='EP Path:')
        self.path_ep.on_change('value', self._textInput)

        self.select_config = Select(title='Configs:', value=None, options=[])
        self.select_config.on_change('value', self._select_config)

        self.button_plot = Button(label='Plot')
        self.button_plot.on_click(self._button_plot)


        tab01 = Panel(child=column(self.div_01, self.path_ep, self.select_config, self.button_plot), title='EP Config')

        return tab01

    def tab_02(self):
        self.k15_individual = FFP()

        self.div_02_01 = Div(text='Footprint por intervalo de 30 minutos através da metodologia Kljun et al. (2015)', width=500)

        self.datetime_slider = DateSlider(title='Datetime:',
                                          start=dt.datetime(2018,1,1,0,0),
                                          end=dt.datetime(2018,1,1,0,30),
                                          value=dt.datetime(2018,1,1,0,0),
                                          step=1000*60*30, format='%x %X')
        self.datetime_slider.on_change('value_throttled', lambda attr, old, new: self.update_ffp())

        self.div_02_02 = Div(text='Selecione os dados', width=500)

        self.source_01 = ColumnDataSource(data=dict(xrs=[], yrs=[]))

        self.fig_01 = figure(title='Footprint K15', plot_height=500, plot_width=500)
        self.fig_01.x_range = Range1d(-1000, 1000)
        self.fig_01.y_range = Range1d(-1000, 1000)

        mlines = self.fig_01.multi_line(xs='xrs', ys='yrs', source=self.source_01)

        tab02 = Panel(child=column(self.div_02_01, self.datetime_slider, self.div_02_02,self.fig_01), title='Footprint per time')

        return tab02

    def tab_03(self):
        self.k15_climatology = FFP_climatology()

        self.div_03 = Div(text='Footprint acumulado através da metodologia Kljun et al. (2015) e direção do vento', width=500)

        self.date_range = DateRangeSlider(title='Date', start=dt.datetime(2018,1,1),
                                          end=dt.datetime(2019,1,1),
                                          value=(dt.datetime(2018,1,1), dt.datetime(2019,1,1)),
                                          step=24*60*60*1000, format="%d/%m/%Y")

        self.time_range = DateRangeSlider(title='Time', start=dt.datetime(2012,1,1,0,0),
                                          end=dt.datetime(2012,1,1,23,30),
                                          value=(dt.datetime(2012,1,1,0,0), dt.datetime(2012,1,1,0,30)),
                                          step=30*60*1000, format='%H:%M')

        self.date_range.on_change('value', lambda attr,old,new:self.update_windDir())
        self.time_range.on_change('value', lambda attr,old,new:self.update_windDir())

        self.button_update_ffp = Button(label='Update Plot', width=500)
        self.button_update_ffp.on_click(self._button_update_ffp)


        self.source_02 = ColumnDataSource(data=dict(xrs=[], yrs=[]))
        self.fig_02 = figure(title='Footprint K15 acumulado', plot_height=500, plot_width=500)
        self.fig_02.x_range = Range1d(-1000, 1000)
        self.fig_02.y_range = Range1d(-1000, 1000)
        mlines = self.fig_02.multi_line(xs='xrs', ys='yrs', source=self.source_02)

        self.source_03 = ColumnDataSource(data=dict(inner=[0], outer=[1], start=[0],end=[2]))
        self.fig_03 = figure(title='Direção do vento', plot_height=500, plot_width=500)
        wedge = self.fig_03.annular_wedge(x=0, y=0, inner_radius='inner', outer_radius='outer', start_angle='start', end_angle='end', color='#FF00FF', source=self.source_03)
        self.date_range.on_change('value', lambda attr,old,new:self.update_windDir())
        self.time_range.on_change('value', lambda attr,old,new:self.update_windDir())
        tab03 = Panel(child=column(self.div_03,
                                   self.date_range,
                                   self.time_range,
                                   self.button_update_ffp,
                                   row(self.fig_02, self.fig_03)), title='Footprint per range')
        return tab03

    def _textInput(self, attr, old, new):
        if self.tabs.active == 0:
            try:
                self.folder_path_ep = pathlib.Path(new)
                readme = self.folder_path_ep.rglob('Readme.txt')
                readme_df = pd.read_csv(list(readme)[0], delimiter=',')
                temp_list = [row.to_list() for i,row in readme_df[['rotation', 'lowfrequency', 'highfrequency','wpl','flagging','name']].iterrows()]
                a = []
                for i in temp_list:
                    a.append('Rotation:{} |LF:{} |HF:{} |WPL:{} |Flag:{}| Name:{}'.format(i[0],i[1],i[2],i[3],i[4],i[5]))
                self.select_config.options = a
            except:
                print('erro text input readme')

    def _select_config(self, attr, old, new):
        print(new)

        full_output_files = self.folder_path_ep.rglob('*{}*_full_output*.csv'.format(new[-3:]))
        dfs_single_config = []
        for file in full_output_files:
            dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999,
                                                 parse_dates={'TIMESTAMP':['date', 'time']},
                                                 keep_date_col=True,
                                                 usecols=self.ep_columns_filtered))
        self.df_ep = pd.concat(dfs_single_config)
        print('ok')

    def update_ffp(self):
        if self.tabs.active == 1:
            datetime = dt.datetime.utcfromtimestamp(self.datetime_slider.value/1000)

            inputs_to_k15 = self.df_ep.loc[self.df_ep['TIMESTAMP']==datetime, ['u_rot', 'L', 'u*','v_var','wind_dir_compass']]
            print(inputs_to_k15)
            self.div_02_02.text = '''
            <table border="2"><tbody><tr><td>&nbsp;zm</td><td>umean</td><td>h</td><td>ol</td><td>sigmav</td><td>ustar</td><td>wind_dir_compass</td>
    		</tr><tr>
    			<td>&nbsp;{}</td><td>{}&nbsp;</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>&nbsp;{}</td>
    		</tr></tbody></table>'''.format(9,
                                      inputs_to_k15['u_rot'].values[0],
                                      1000,
                                      inputs_to_k15['L'].values[0],
                                      inputs_to_k15['v_var'].values[0],
                                      inputs_to_k15['u*'].values[0],
                                      inputs_to_k15['wind_dir_compass'].values[0])
            out = self.k15_individual.output(zm=9,
                                             umean=inputs_to_k15['u_rot'].values[0],
                                             h=1000,
                                             ol=inputs_to_k15['L'].values[0],
                                             sigmav=inputs_to_k15['v_var'].values[0],
                                             ustar=inputs_to_k15['u*'].values[0],
                                             wind_dir=inputs_to_k15['wind_dir_compass'].values[0],
                                             rs=[0.3, 0.9], crop=False, fig=False)
            self.source_01.data = dict(xrs=out[8], yrs=out[9])

        if self.tabs.active == 2:
            # try:
            start_date = dt.datetime.utcfromtimestamp(self.date_range.value[0]/1000).date()
            end_date = dt.datetime.utcfromtimestamp(self.date_range.value[1]/1000).date()

            start_time = dt.datetime.utcfromtimestamp(self.time_range.value[0]/1000).time()
            end_time = dt.datetime.utcfromtimestamp(self.time_range.value[1]/1000).time()

            inputs_to_k15 = self.df_ep.loc[
                (self.df_ep['TIMESTAMP'].dt.date >= start_date) &
                (self.df_ep['TIMESTAMP'].dt.date <= end_date) &
                (self.df_ep['TIMESTAMP'].dt.time >= start_time) &
                (self.df_ep['TIMESTAMP'].dt.time <= end_time),
                ['u_rot','L','u*', 'v_var','wind_dir_compass']
            ]

            out = self.k15_climatology.output(zm=9,
                                              umean=inputs_to_k15['u_rot'].to_list(),
                                              h=[1000 for i in range(len(inputs_to_k15['u_rot'].to_list()))],
                                              ol=inputs_to_k15['L'].to_list(),
                                              sigmav=inputs_to_k15['v_var'].to_list(),
                                              ustar=inputs_to_k15['u*'].to_list(),
                                              wind_dir=inputs_to_k15['wind_dir_compass'].to_list(),
                                              rs=[0.3, 0.9], crop=False, fig=False)
            self.source_02.data = dict(xrs=out['xr'], yrs=out['yr'])
            # except:
            #     print('erro update')

    def update_windDir(self):
        start_date = dt.datetime.utcfromtimestamp(self.date_range.value[0]/1000).date()
        end_date = dt.datetime.utcfromtimestamp(self.date_range.value[1]/1000).date()

        start_time = dt.datetime.utcfromtimestamp(self.time_range.value[0]/1000).time()
        end_time = dt.datetime.utcfromtimestamp(self.time_range.value[1]/1000).time()

        start_angle = np.arange(0,360,10)*np.pi/180 + 90*np.pi/180
        end_angle = np.arange(10,370,10)*np.pi/180 + 90*np.pi/180

        filter = self.df_ep[(self.df_ep['TIMESTAMP'].dt.date >= start_date) &
                            (self.df_ep['TIMESTAMP'].dt.date <= end_date) &
                            (self.df_ep['TIMESTAMP'].dt.time >= start_time) &
                            (self.df_ep['TIMESTAMP'].dt.time <= end_time)]

        self.source_03.data = dict(inner=[0 for i in range(36)],
                                   outer=filter.groupby(by='wind_bin').count()['wind_dir_compass'][::-1]/filter.groupby(by='wind_bin').count()['wind_dir_compass'].max(),
                                   start=start_angle,
                                   end=end_angle)
        print(filter.groupby(by='wind_bin').count()['wind_dir_compass'])

    def _button_plot(self):
        self.datetime_slider.start = self.df_ep['TIMESTAMP'].min()
        self.datetime_slider.end = self.df_ep['TIMESTAMP'].max()
        self.datetime_slider.value = self.df_ep['TIMESTAMP'].min()

        # self.date_range.start = self.df_ep['date'].min()
        # self.date_range.end = self.df_ep['date'].max()
        # self.date_range.value = (self.df_ep['date'].min(), self.df_ep['date'].max())

        self._adjust_wind_direction()

        self.theta = np.linspace(0, 360, 36)
        theta01 = np.linspace(0, 360, 37)
        # print(theta01)
        self.df_ep['wind_bin'] = pd.cut(x=self.df_ep['wind_dir_compass'], bins=theta01)
        # print(self.df_ep['wind_bin'])

        self.date_range.start = self.df_ep['date'].min()
        self.date_range.end = self.df_ep['date'].max()
        self.date_range.value = (self.df_ep['date'].min(), self.df_ep['date'].max())

    def _button_update_ffp(self):
        self.update_ffp()


    def _adjust_wind_direction(self):
        self.df_ep['wind_dir_sonic'] = 360 - self.df_ep['wind_dir']
        azimute = 135.1
        self.df_ep['wind_dir_compass'] = (360 + azimute - self.df_ep['wind_dir_sonic']).apply(lambda x: x-360 if x>=360 else x)


view_k15()
