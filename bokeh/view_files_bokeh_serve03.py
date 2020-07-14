from bokeh.io import show, output_file, curdoc
from bokeh.models import Button, TextInput, Paragraph, Select, Panel, Tabs,ColumnDataSource, RangeTool, Circle, Slope, Label, Legend, LegendItem, LinearColorMapper, Div,CheckboxButtonGroup,Slider,CheckboxGroup,RangeSlider,BasicTicker, ColorBar,DatetimeTickFormatter
from bokeh.layouts import gridplot, column, row
from bokeh.plotting import figure
from bokeh.transform import transform

from bokeh.models import DateRangeSlider


import pathlib, sys
sys.path.append(str(pathlib.Path(r'C:\Users\User\Mestrado\Penman_monteith')))
from radiation02 import Radiation

import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import linregress

class view_files:
    def __init__(self):
        # self.radiation = Radiation()
        print('entrou')
        self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
                                     'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed',
                                     'wind_dir', 'u*', '(z-d)/L',  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var','H_strg','LE_strg']

        self.lf_columns_filtered = ['TIMESTAMP','Hs','u_star','Ts_stdev','Ux_stdev','Uy_stdev','Uz_stdev','Ux_Avg', 'Uy_Avg', 'Uz_Avg',
                                    'Ts_Avg', 'LE_wpl', 'Hc','H2O_mean', 'amb_tmpr_Avg', 'amb_press_mean', 'Tc_mean', 'rho_a_mean','CO2_sig_strgth_mean',
                                    'H2O_sig_strgth_mean','T_tmpr_rh_mean', 'e_tmpr_rh_mean', 'e_sat_tmpr_rh_mean', 'H2O_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                     'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
                                      'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)', 'precip_Tot', 'panel_tmpr_Avg']


        self.TOOLS_01 = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

        self.tabs = Tabs(tabs=[self.tab_01(), self.tab_02()])



        curdoc().add_root(row(self.sidebar_01(),self.tabs))

    def sidebar_01(self):
        self.p01 = Div(text=r"""C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01""", width=100,style={'font-size': '75%'})

        self.path = TextInput(value='', title='Insert EP Path:')
        self.path.on_change('value', self._textInput_EP)

        self.select_config = Select(title='Select EP Configs:', value=None, options=[])
        self.select_config.on_change('value', self._select_config_EP)

        self.p02 = Div(text=r"""C:\Users\User\Mestrado\Dados_brutos""", width=100,style={'font-size': '75%'})

        self.path2 = TextInput(value='', title='Insert LF Path:')
        self.path2.on_change('value', self._textInput_LF)

        self.html_lf = Div(text='Sem dados', width=300)

        self.button_plot = Button(label='Plot')
        self.button_plot.on_click(self._button_plot)

        self.slider_signalStrFilter = Slider(start=0, end=1, value=0, step=0.01, title='Signal Strength:')

        self.checkbox_flag = CheckboxButtonGroup(labels=['0', '1', '2'], active=[0,1,2])
        self.checkbox_flag.on_click(self._button_plot_click)

        self.checkbox_rain = CheckboxGroup(labels=['Rain Filter'])
        self.checkbox_rain.on_click(self._button_plot_click)

        self.daterangeslider = DateRangeSlider(title='Date', start=dt.datetime(2018,1,1),end=dt.datetime(2019,1,1), value=(dt.datetime(2017, 9, 7), dt.datetime(2017, 10, 15)), step=24*60*60*1000, format="%d/%m/%Y")

        self.timerangeslider = DateRangeSlider(title='Time', start=dt.datetime(2012,1,1,0,0),end=dt.datetime(2012,1,1,23,30), value=(dt.datetime(2012,1,1,0,0), dt.datetime(2012,1,1,0,30)),step=30*60*1000, format='%H:%M')

        self.text_save_df = TextInput(value='', title='Insert path to save')
        self.save_df = Button(label='Save Dataframe')
        self.save_df.on_click(self._button_save)


        controls = [self.slider_signalStrFilter, self.daterangeslider, self.timerangeslider]
        for control in controls:
            control.on_change('value_throttled', lambda attr, old, new:self.update_01())

        return column(self.p01,
                      self.path,
                      self.select_config,
                      self.p02,
                      self.path2,
                      self.html_lf,
                      self.button_plot,
                      self.slider_signalStrFilter,
                      Div(text='Mauder and Foken Flag:'),
                      self.checkbox_flag,
                      self.checkbox_rain,
                      self.daterangeslider,
                      self.timerangeslider,
                      self.text_save_df,
                      self.save_df)

    def tab_01(self):

        self.source_01 = ColumnDataSource(data=dict(x=[], y=[], y02=[], date=[], time=[], ET=[]))

        self.source_02 = ColumnDataSource(data=dict(x=[], y=[],text01=[]))

        # Figure 01 (EP)
        self.fig_01 = figure(title='EP', plot_height=250, plot_width=700, x_axis_type='datetime', tools=self.TOOLS_01)
        circle_ep = self.fig_01.circle(x='x', y='y', source=self.source_01,color='blue',
                                       nonselection_fill_alpha=0.1,nonselection_fill_color="grey",nonselection_line_color="grey",nonselection_line_alpha=0.1)
        #####

        # Figure 02 (LF)
        self.fig_02 = figure(title='LF', plot_height=250, plot_width=700, x_axis_type='datetime', x_range=self.fig_01.x_range,tools=self.TOOLS_01)
        circle_lf = self.fig_02.circle(x='x', y='y02', source=self.source_01, color='red',
                                       nonselection_fill_alpha=0.1,nonselection_fill_color="grey",nonselection_line_color="grey",nonselection_line_alpha=0.1)
        circle_lf.data_source.selected.on_change('indices', self._selection_energybalance)
        #####

        # Figure 03 (EP x LF)
        self.fig_03 = figure(title='EP x LF', plot_height=500, plot_width=500)
        circle_ep_lf = self.fig_03.circle(x='y02', y='y', source=self.source_01,
                                          color='#7A2DBD', selection_color="#7A2DBD", selection_fill_alpha=0.3, selection_line_alpha=0.5,
                                          nonselection_fill_alpha=0.1,nonselection_fill_color="grey",nonselection_line_color="grey",nonselection_line_alpha=0.1)

        slope_11 = Slope(gradient=1, y_intercept=0, line_color='orange', line_dash='dashed', line_width=3)
        self.fig_03.add_layout(slope_11)

        self.slope_fit = Slope(gradient=1, y_intercept=0, line_color='red', line_width=3)
        self.fig_03.add_layout(self.slope_fit)

        self.text01 = self.fig_03.text(x='x',y='y',text='text01', source=self.source_02,text_baseline='top')
        ######

        # Figure 04 (ET)
        self.fig_04 = figure(title='ET', plot_height=350, plot_width=1200, x_axis_type='datetime', y_axis_type='datetime', tools="hover,pan,wheel_zoom,box_zoom,reset,box_select,tap")
        self.fig_04.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_04.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])

        colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
        self.color_mapper = LinearColorMapper(palette=colors)

        self.et = self.fig_04.rect(x='date', y='time', fill_color=transform('ET', self.color_mapper), source=self.source_01, width=1000*60*60*24, height=1000*60*30, line_color=None)
        color_bar = ColorBar(color_mapper=self.color_mapper, ticker=BasicTicker(desired_num_ticks=len(colors)),label_standoff=6, border_line_color=None, location=(0,0))
        self.fig_04.add_layout(color_bar, 'right')

        self.fig_04.hover.tooltips = [
                                       ("ET (mm/h)", "@ET")
        ]

        # self.et.xaxis[0]
        self.fig_04.axis.axis_line_color = None
        self.fig_04.axis.major_tick_line_color = None


        self.fig_04.xaxis.major_label_orientation = 1

        tab01 = Panel(child=column(row(column(self.fig_01,
                                              self.fig_02), self.fig_03),
                                   self.fig_04), title='Energy Balance')
        return tab01

    def tab_02(self):
        self.select_lf_column = Select(title='LF Radiation Column', value=None, options=[])
        self.select_lf_column.on_change('value', self._select_lf_column)
        self.source_03 = ColumnDataSource(data=dict(x=[], y=[], rad=[]))

        self.fig_05 = figure(title='LF radiation', plot_height=350, plot_width=1200, x_axis_type='datetime')
        radiation_circle = self.fig_05.circle(x='x',y='y', source=self.source_03)

        erad = self.fig_05.circle(x='x', y='rad', source=self.source_03, color='red')

        tab02 = Panel(child=column(self.select_lf_column,self.fig_05), title='Radiation')

        return tab02

    def _textInput_EP(self, attr, old,new):
        try:
            print(attr)
            self.folder_path_ep = pathlib.Path(new)
            readme = self.folder_path_ep.rglob('Readme.txt')
            readme_df = pd.read_csv(list(readme)[0], delimiter=',')
            temp_list = [row.to_list() for i,row in readme_df[['rotation', 'lowfrequency','highfrequency','wpl','flagging','name']].iterrows()]
            a = []
            for i in temp_list:
                a.append('Rotation:{} |LF:{} |HF:{} |WPL:{} |Flag:{}| Name:{}'.format(i[0],i[1],i[2],i[3],i[4],i[5]))
            self.select_config.options = a
        except:
            print('erro')

    def _textInput_LF(self, attr, old, new):
        try:
            self.folder_path_lf = pathlib.Path(new)
            lf_files = self.folder_path_lf.rglob('TOA5*.flux.dat')
            self.dfs_02_01 = []
            for file in lf_files:
                self.dfs_02_01.append(pd.read_csv(file, skiprows=[0,2,3], parse_dates=['TIMESTAMP'],na_values='NAN', usecols=self.lf_columns_filtered))

            self.dfs_concat_02_01 = pd.concat(self.dfs_02_01)

            self.html_lf.text = "<table> <tr><td><span style='font-weight:bold'>Number of Files:</spam></td> <td>{}</td></tr><tr><td><span style='font-weight:bold'>Begin:</span></td> <td>{}</td></tr> <tr> <td><span style='font-weight:bold'>End:</span></td><td>{}</td>  </tr>".format(len(self.dfs_02_01), self.dfs_concat_02_01['TIMESTAMP'].min(),self.dfs_concat_02_01['TIMESTAMP'].max())

        except:
            self.html_lf.text = 'Erro, insira outro Path'
            print('erro2')

    def _select_config_EP(self, attr, old, new):
        print(new)

        full_output_files = self.folder_path_ep.rglob('*{}*_full_output*.csv'.format(new[-3:]))
        dfs_single_config = []
        for file in full_output_files:
            dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']},keep_date_col=True, usecols=self.ep_columns_filtered))

        self.df_ep = pd.concat(dfs_single_config)
        print('ok')

    def _button_plot(self):
        self.dfs_compare = pd.merge(left=self.dfs_concat_02_01, right=self.df_ep, how='outer', on='TIMESTAMP', suffixes=("_lf","_ep"))

        self.daterangeslider.start = self.dfs_compare['TIMESTAMP'].dt.date.min()
        self.daterangeslider.end = self.dfs_compare['TIMESTAMP'].dt.date.max()
        self.daterangeslider.value = (self.dfs_compare['TIMESTAMP'].dt.date.min(), self.dfs_compare['TIMESTAMP'].dt.date.max())

        self.dfs_compare['date_ns'] = pd.to_datetime(self.dfs_compare['date'])
        self.dfs_compare['time_ns'] = pd.to_datetime(self.dfs_compare['time'])

        self.select_lf_column.options = ['TIMESTAMP', 'amb_tmpr_Avg', 'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
         'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)', 'e', 'es']

        self.radiation = Radiation(lat_degree=-23, date=self.dfs_compare['TIMESTAMP'].dt.date.to_list())

        self.dfs_compare_copy = self.dfs_compare.copy()

        # print(self.radiation.output())
        # self.lf_columns_filtered = ['TIMESTAMP','Hs','u_star','Ts_stdev','Ux_stdev','Uy_stdev','Uz_stdev','Ux_Avg', 'Uy_Avg', 'Uz_Avg',
        #                             'Ts_Avg', 'LE_wpl', 'Hc','H2O_mean', 'amb_tmpr_Avg','Tc_mean'
        #                              'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
        #                               'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)']
        self.update_01()

    def filter_flag(self):
        try:
            start_date = dt.datetime.utcfromtimestamp(self.daterangeslider.value[0]/1000).date()
            end_date = dt.datetime.utcfromtimestamp(self.daterangeslider.value[1]/1000).date()

            start_time = dt.datetime.utcfromtimestamp(self.timerangeslider.value[0]/1000).time()
            end_time = dt.datetime.utcfromtimestamp(self.timerangeslider.value[1]/1000).time()

            flag = self.dfs_compare[
                (self.dfs_compare['H2O_sig_strgth_mean'] >= self.slider_signalStrFilter.value) &
                (self.dfs_compare['TIMESTAMP'].dt.date >= start_date) &
                (self.dfs_compare['TIMESTAMP'].dt.date <= end_date) &
                (self.dfs_compare['TIMESTAMP'].dt.time >= start_time) &
                (self.dfs_compare['TIMESTAMP'].dt.time <= end_time)
            ]

            # self.dfs_compare_copy.loc[
            #     (self.dfs_compare_copy['H2O_sig_strgth_mean'] >= self.slider_signalStrFilter.value) &
            #     (self.dfs_compare_copy['TIMESTAMP'].dt.date >= start_date) &
            #     (self.dfs_compare_copy['TIMESTAMP'].dt.date <= end_date) &
            #     (self.dfs_compare_copy['TIMESTAMP'].dt.time >= start_time) &
            #     (self.dfs_compare_copy['TIMESTAMP'].dt.time <= end_time), 'ET'] = np.nan
            # self.dfs_compare_copy.loc[(self.dfs_compare_copy['H2O_sig_strgth_mean'] <= self.slider_signalStrFilter.value)] = np.nan
        except:
            print('erro')
            flag = self.dfs_compare[
                (self.dfs_compare['H2O_sig_strgth_mean'] >= self.slider_signalStrFilter.value)
            ]
            self.dfs_compare_copy.loc[
                (self.dfs_compare_copy['H2O_sig_strgth_mean'] <= self.slider_signalStrFilter.value), 'ET'] = np.nan

        if self.checkbox_rain.active == [0]:
            flag = flag[flag['precip_Tot']==0]
            self.dfs_compare_copy.loc[(self.dfs_compare_copy['precip_Tot']!=0), 'ET'] = np.nan

        flag = flag[flag[['qc_H','qc_LE']].isin(self.checkbox_flag.active).sum(axis=1)==2]

        self.dfs_compare_copy.loc[self.dfs_compare_copy[['qc_H','qc_LE']].isin(self.checkbox_flag.active).sum(axis=1)!=2,'ET'] = np.nan

        # print(self.dfs_compare_copy.loc[self.dfs_compare_copy['qc_H']==2, 'ET'])


        return flag

    def update_01(self):
        self.df_filter = self.filter_flag()

        self.source_01.data = dict(x=self.df_filter['TIMESTAMP'],
                                   y=self.df_filter[['H', 'LE','H_strg','LE_strg']].sum(axis=1, min_count=1),
                                   y02=self.df_filter['Rn_Avg']-self.df_filter[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1),
                                   date=self.df_filter['date_ns'],
                                   time=self.df_filter['time_ns'],
                                   ET=self.df_filter['ET'])

        self.color_mapper.low = self.df_filter['ET'].min()
        self.color_mapper.high = self.df_filter['ET'].max()

        self.fig_01.xaxis.axis_label = 'TIMESTAMP'
        self.fig_01.yaxis.axis_label = 'H + LE (W m-2)'

        self.fig_02.xaxis.axis_label = 'TIMESTAMP'
        self.fig_02.yaxis.axis_label = 'Rn - G (W m-2)'

        self.fig_03.xaxis.axis_label = 'Rn - G (W m-2)'
        self.fig_03.yaxis.axis_label = 'H + LE (W m-2)'

        self.df_corr = pd.DataFrame()
        self.df_corr['EP'] = self.df_filter[['H','LE','H_strg','LE_strg']].sum(axis=1, min_count=1)
        self.df_corr['LF'] = self.df_filter['Rn_Avg'] - self.df_filter[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1)

        pearson = self.df_corr.corr(method='pearson')['LF'][0]
        self.df_corr.dropna(inplace=True)
        linear_regression = linregress(x=self.df_corr['LF'], y=self.df_corr['EP'])

        x = np.array(self.df_corr['LF'].to_list())
        x1 = x[:, np.newaxis]
        fit_linear = np.linalg.lstsq(x1, self.df_corr['EP'], rcond=None)[0][0]

        self.slope_fit.gradient = fit_linear

        self.source_02.data = dict(x=[np.nanmin(self.df_corr['LF'])],
                                   y=[np.nanmax(self.df_corr['EP'])],
                                   text01=['Pearson: {:.4f}\nSlope: {:.4f}\ny = {:.4f}x + {:.4f}'.format(pearson, fit_linear,linear_regression[0],linear_regression[1])])



    def _button_plot_click(self, new):
        self.update_01()

    def _selection_energybalance(self, attr, old, new):
        try:
            df_selected = self.df_filter[['Rn_Avg','H','LE','H_strg','LE_strg','shf_Avg(1)','shf_Avg(2)']].iloc[self.source_01.selected.indices]

            df_selected_corr = pd.DataFrame()
            df_selected_corr['EP'] = df_selected[['H','LE','H_strg','LE_strg']].sum(axis=1, min_count=1)
            df_selected_corr['LF'] = df_selected['Rn_Avg'] - df_selected[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1)
            df_selected_corr.dropna(inplace=True)

            x = np.array(df_selected_corr['LF'].to_list())
            x1 = x[:, np.newaxis]
            fit_linear = np.linalg.lstsq(x1, df_selected_corr['EP'], rcond=None)[0][0]

            pearson_corr = df_selected_corr.corr(method='pearson')['LF'][0]
            linear_regression = linregress(x=df_selected_corr['LF'], y=df_selected_corr['EP'])

            self.source_02.data = dict(x=[np.nanmin(self.df_corr['LF'])],
                                       y=[np.nanmax(self.df_corr['EP'])],
                                       text01=['Pearson: {:.4f}\nSlope: {:.4f}\ny = {:.4f}x + {:.4f}'.format(pearson_corr, fit_linear,linear_regression[0],linear_regression[1])])
            self.slope_fit.gradient = fit_linear

            print(pearson_corr)
            print(fit_linear)
            print(linear_regression[0],linear_regression[1])
        except:
            print('sem selecao')

    def _select_lf_column(self, attr, old, new):
        print(self.select_lf_column.value)

        self.fig_05.xaxis.axis_label = 'TIMESTAMP'
        self.fig_05.yaxis.axis_label = self.select_lf_column.value
        self.source_03.data = dict(x=self.dfs_compare['TIMESTAMP'],
                                   y=self.dfs_compare[self.select_lf_column.value],
                                   rad=self.radiation.output()[1]*11.57)


    def _button_save(self):
        print(self.text_save_df.value)
        path = pathlib.Path('{}'.format(self.text_save_df.value))
        self.df_filter.to_csv(path/'df_filter.csv')
        self.dfs_compare_copy.to_csv(path/'dfs_compare.csv')

view_files()
