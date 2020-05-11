import ipywidgets
import pandas as pd
import numpy as np
import pathlib
import datetime as dt
# import matplotlib.pyplot as plt
from bqplot import pyplot as plt
import bqplot as bq
import functools


class Viewer:
    def __init__(self):
        self.time_interval_01 = 500

        self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
 'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',
  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var']
        # self.ep_columns_all =

        self.lf_columns_filtered = ['TIMESTAMP','Hs','u_star','Ts_stdev','Ux_stdev','Uy_stdev','Uz_stdev','Ux_Avg', 'Uy_Avg', 'Uz_Avg',
                                    'Ts_Avg', 'LE_wpl', 'Hc','H2O_mean', 'amb_tmpr_Avg', 'amb_press_mean', 'Tc_mean', 'rho_a_mean','CO2_sig_strgth_mean',
                                    'H2O_sig_strgth_mean','T_tmpr_rh_mean', 'e_tmpr_rh_mean', 'e_sat_tmpr_rh_mean', 'H2O_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                     'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
                                      'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)', 'precip_Tot',
 'panel_tmpr_Avg',]

        self.tabs = ipywidgets.Tab([self.tab00(), self.tab01(), self.tab02(), self.tab03(), self.tab04()])


        self.tabs.set_title(0, 'EP - Master Folder')
        self.tabs.set_title(1, 'EP - Simple View')
        self.tabs.set_title(2, 'LowFreq - Master Folder')
        self.tabs.set_title(3, 'LowFreq - Simple View')
        self.tabs.set_title(4, 'Compare - EP/LF')
        display(self.tabs)

    def tab00(self):
        self.out_00 = ipywidgets.Output()
        with self.out_00:
            self.text_metafile_00_00 = ipywidgets.Text(placeholder='Insert the Metafile Path',
                                                       layout=ipywidgets.Layout(width='90%'))

            self.button_addPathMeta_00_00 = ipywidgets.Button(description='Apply')
            self.button_addPathMeta_00_00.on_click(self._button_addPathMeta)

            self.selectMultiple_Meta_00_00 = ipywidgets.SelectMultiple(description='Configurations:',
                                                                       layout=ipywidgets.Layout(width='90%'),
                                                                       style={'description_width':'initial'})
            self.selectMultiple_Meta_00_00.observe(self._selectMultiple_config, 'value')



        return ipywidgets.VBox([ipywidgets.HBox([self.text_metafile_00_00, self.button_addPathMeta_00_00]),
                                self.selectMultiple_Meta_00_00,
                                self.out_00])

    def tab01(self):
        self.out_01 = ipywidgets.Output()
        with self.out_01:


            self.dropdown_xAxis_01_01 = ipywidgets.Dropdown(description='X-Axis')
            self.dropdown_yAxis_01_01 = ipywidgets.Dropdown(description='Y-Axis')

            self.button_plot_01_01 = ipywidgets.Button(description='Plot')
            self.button_plot_01_01.on_click(self._button_plot)


            self.accordion_01 = ipywidgets.Accordion()


            self.x_scale_01_01 = bq.DateScale()
            self.y_scale_01_01 = bq.LinearScale()
            self.x_axis_01_01 = bq.Axis(scale=self.x_scale_01_01)
            self.y_axis_01_01 = bq.Axis(scale=self.y_scale_01_01, orientation='vertical')
            self.fig_01_01 = bq.Figure(axes=[self.x_axis_01_01, self.y_axis_01_01],
                                       animation_duration=self.time_interval_01)



            self.x_scale_01_02 = bq.DateScale()
            self.y_scale_01_02 = bq.LinearScale()
            self.x_axis_01_02 = bq.Axis(scale=self.x_scale_01_02)
            self.y_axis_01_02 = bq.Axis(scale=self.y_scale_01_02, orientation='vertical')
            self.fig_01_02 = bq.Figure(axes=[self.x_axis_01_02, self.y_axis_01_02],
                                       animation_duration=self.time_interval_01)

            self.selectionSlider_flag = ipywidgets.SelectionSlider(options=[0,1,2,'All'],
                                                                    value='All',
                                                                    description='Flag')
            self.selectionSlider_flag.observe(self._selectionSlider_flag, 'value')

            self.accordion_01.children = [ipywidgets.VBox([self.fig_01_01]),
                                          ipywidgets.VBox([self.selectionSlider_flag,
                                                           self.fig_01_02])]

            self.accordion_01.set_title(0, 'Simple Plot')
            self.accordion_01.set_title(1, 'Flag Plot')

            self.accordion_01.selected_index = None



        return ipywidgets.VBox([ipywidgets.HBox([self.dropdown_xAxis_01_01, self.dropdown_yAxis_01_01]),
                                self.button_plot_01_01,
                                self.accordion_01,
                                self.out_01])

    def tab02(self):
        self.out_02 = ipywidgets.Output()
        with self.out_02:
            self.text_lowfreqfile_02_00 = ipywidgets.Text(placeholder='Insert the LowFreq Files Path',
                                                          layout=ipywidgets.Layout(width='90%'))

            self.button_view_lowFreq_02_00 = ipywidgets.Button(description='Apply')
            self.button_view_lowFreq_02_00.on_click(self._button_viewLowFreq)

            self.html_02_01 = ipywidgets.HTML()



        return ipywidgets.VBox([ipywidgets.HBox([self.text_lowfreqfile_02_00, self.button_view_lowFreq_02_00]),
                                self.html_02_01,
                                self.out_02])


    def tab03(self):
        self.out_03 = ipywidgets.Output()
        with self.out_03:
            self.dropdown_xAxis_03_01 = ipywidgets.Dropdown(description='X-Axis')
            self.dropdown_yAxis_03_01 = ipywidgets.Dropdown(description='Y-Axis')

            self.button_plot_03_01 = ipywidgets.Button(description='Plot')
            self.button_plot_03_01.on_click(self._button_plot)

            self.x_scale_03_01 = bq.DateScale()
            self.y_scale_03_01 = bq.LinearScale()

            self.x_axis_03_01 = bq.Axis(scale=self.x_scale_03_01)
            self.y_axis_03_01 = bq.Axis(scale=self.y_scale_03_01, orientation='vertical')

            self.fig_03_01 = bq.Figure(axes=[self.x_axis_03_01, self.y_axis_03_01],
                                       animation_duration=self.time_interval_01)

        return ipywidgets.VBox([ipywidgets.HBox([self.dropdown_xAxis_03_01, self.dropdown_yAxis_03_01]),
                                self.button_plot_03_01,
                                self.fig_03_01,
                                self.out_03])

    def tab04(self):
        self.out_04 = ipywidgets.Output()
        with self.out_04:
            self.html_04_01 = ipywidgets.HTML(value="<p>Required input at <b>EP - Master folder</b> and&nbsp;<b>LowFreq - Master folder</b>.</p>")

            self.dropdown_xAxis_04_01 = ipywidgets.Dropdown(description='X-Axis (EP)')
            self.dropdown_xAxis_04_02 = ipywidgets.Dropdown(description='X-Axis (LF)')

            self.selectMultiple_04_01 = ipywidgets.SelectMultiple(description='Y-Axis (EP)')
            self.selectMultiple_04_02 = ipywidgets.SelectMultiple(description='Y-Axis (LF)')

            self.selectionSlider_flag.observe(self._selectionSlider_flag, 'value')

            self.floatSlider_signalStr = ipywidgets.FloatSlider(min=0, max=1, step=0.01, value=0,
                                                                description='Strength Signal',
                                                                continuous_update=False, redout_format='.2f')
            self.floatSlider_signalStr.observe(self._floatSlider_signalStr, 'value')

            self.checkbox_rain = ipywidgets.Checkbox(value=False, description='Filter Precipitation Periods')

            self.button_plot_04_01 = ipywidgets.Button(description='Plot')
            self.button_plot_04_01.on_click(self._button_plot)


            self.x_scale_04_01 = bq.DateScale()
            self.y_scale_04_01 = bq.LinearScale()
            self.x_axis_04_01 = bq.Axis(scale=self.x_scale_04_01)
            self.y_axis_04_01 = bq.Axis(scale=self.y_scale_04_01, orientation='vertical')
            self.fig_04_01 = bq.Figure(axes=[self.x_axis_04_01, self.y_axis_04_01])



            self.x_scale_04_02 = bq.LinearScale()
            self.y_scale_04_02 = bq.LinearScale()
            self.x_axis_04_02 = bq.Axis(scale=self.x_scale_04_02)
            self.y_axis_04_02 = bq.Axis(scale=self.y_scale_04_02, orientation='vertical')
            self.fig_04_02 = bq.Figure(axes=[self.x_axis_04_02, self.y_axis_04_02])


            self.brushintsel = bq.interacts.BrushIntervalSelector(scale=self.x_scale_04_01)
            self.brushintsel.observe(self._update,'selected')


        return ipywidgets.VBox([self.html_04_01,
                                ipywidgets.HBox([self.dropdown_xAxis_04_01, self.selectMultiple_04_01]),
                                ipywidgets.HBox([self.dropdown_xAxis_04_02, self.selectMultiple_04_02]),
                                self.selectionSlider_flag,
                                self.floatSlider_signalStr,
                                self.checkbox_rain,
                                self.button_plot_04_01,
                                self.fig_04_01,
                                self.fig_04_02,
                                self.out_04])

    def _selectMultiple_config(self, *args):
        with self.out_00:
            # print('asd')
            self.scatter_01_01 = []
            self.scatter_01_02 = []


            # self.dfs_01_01 => [[partes para cada config] numero de configs]
            # self.dfs_01_01 => agora é [numero de configs]
            self.dfs_01_01 = []
            for i in self.selectMultiple_Meta_00_00.index:
                dfs_single_config = []
                full_output_files = self.folder_path.rglob('*{}*_full_output*.csv'.format(self.config_name[i]))
                for file in full_output_files:
                    dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']}, usecols=self.ep_columns_filtered))

                dfs_concat_singleconfig = pd.concat(dfs_single_config)
                self.scatter_01_01.append(bq.Scatter(scales={'x':self.x_scale_01_01, 'y':self.y_scale_01_01}))
                self.scatter_01_02.append(bq.Scatter(scales={'x':self.x_scale_01_02, 'y':self.y_scale_01_02}))

                # self.dfs_01_01.append(dfs_single_config)
                self.dfs_01_01.append(dfs_concat_singleconfig)

                # self.dropdown_xAxis_01_01.options = pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']}, usecols=self.ep_columns_filtered).columns.to_list()
                self.dropdown_xAxis_01_01.options = dfs_single_config[0].columns.to_list()
                self.dropdown_yAxis_01_01.options = self.dropdown_xAxis_01_01.options

                self.dropdown_xAxis_04_01.options = self.dropdown_xAxis_01_01.options
                self.selectMultiple_04_01.options = self.dropdown_xAxis_01_01.options

            self.fig_01_01.marks = self.scatter_01_01
            self.fig_01_02.marks = self.scatter_01_02

    def _selectionSlider_flag(self, *args):
        if (self.tabs.selected_index == 1) and (self.accordion_01.selected_index == 1):
            with self.out_01:
                for i,df in enumerate(self.dfs_01_01):
                    with self.scatter_01_02[i].hold_sync():
                        if self.selectionSlider_flag.value == 'All':
                            self.scatter_01_02[i].x = df['{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                            self.scatter_01_02[i].y = df['{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                            self.scatter_01_02[i].colors = ['blue']

                        if self.selectionSlider_flag.value in [0,1,2]:
                            self.scatter_01_02[i].x = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_flag.value,'{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                            self.scatter_01_02[i].y = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_flag.value,'{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                            if self.selectionSlider_flag.value == 0:
                                self.scatter_01_02[i].colors = ['green']
                            if self.selectionSlider_flag.value == 1:
                                self.scatter_01_02[i].colors = ['orange']
                            if self.selectionSlider_flag.value == 2:
                                self.scatter_01_02[i].colors = ['red']

        if (self.tabs.selected_index == 4):
            with self.out_04:
                qc_list = list(map(lambda x: 'qc_'+x,list(self.selectMultiple_04_01.value)))

                for i, df in enumerate(self.dfs_compare):
                    if self.selectionSlider_flag.value == 'All':
                        with self.scatter_04_ep[i].hold_sync():

                            self.flag_filter = df[qc_list].isin([0,1,2]).sum(axis=1)==len(self.selectMultiple_04_01.value)
                            self.str_filter = df['H2O_sig_strgth_mean'] > self.floatSlider_signalStr.value

                            self.scatter_04_ep[i].x = df.loc[self.flag_filter & self.str_filter,'{}'.format(self.dropdown_xAxis_04_01.value)].to_list()
                            self.scatter_04_ep[i].y = df.loc[self.flag_filter & self.str_filter,list(self.selectMultiple_04_01.value)].sum(axis=1, min_count=1).to_list()

                    if self.selectionSlider_flag.value in [0,1,2]:
                        with self.scatter_04_ep[i].hold_sync():

                            self.flag_filter = df[qc_list].isin([self.selectionSlider_flag.value]).sum(axis=1)==len(self.selectMultiple_04_01.value)
                            self.str_filter = df['H2O_sig_strgth_mean'] > self.floatSlider_signalStr.value

                            self.scatter_04_ep[i].x = df.loc[self.flag_filter & self.str_filter,'{}'.format(self.dropdown_xAxis_04_01.value)].to_list()
                            self.scatter_04_ep[i].y = df.loc[self.flag_filter & self.str_filter,list(self.selectMultiple_04_01.value)].sum(axis=1, min_count=1).to_list()

                            # self.scatter_04_ep[i].x = df.loc[df['qc_{}'.format(self.dropdown_yAxis_04_01.value)]==self.selectionSlider_flag.value]

                # if self.selectionSlider_flag.value in ['All',0,1,2]:
                #     with self.scatter_04_lf[0].hold_sync():
                #         print(self.dfs_compare[0][list(self.selectMultiple_04_02.value)].sum(axis=1, min_count=1).to_list())
                #         self.scatter_04_lf[0].x = self.dfs_compare[0][self.dropdown_xAxis_04_02.value].to_list()
                #         self.scatter_04_lf[0].y = self.dfs_compare[0][list(self.selectMultiple_04_02.value)].sum(axis=1, min_count=1).to_list()

                # if self.selectionSlider_flag.value in [0,1,2]:

    def _floatSlider_signalStr(self, *args):
        if (self.tabs.selected_index == 4):
            with self.out_04:
                qc_list = list(map(lambda x: 'qc_'+x,list(self.selectMultiple_04_01.value)))

                for i, df in enumerate(self.dfs_compare):
                    if self.selectionSlider_flag.value == 'All':
                        with self.scatter_04_ep[i].hold_sync():

                            self.flag_filter = df[qc_list].isin([0,1,2]).sum(axis=1)==len(self.selectMultiple_04_01.value)
                            self.str_filter = df['H2O_sig_strgth_mean'] > self.floatSlider_signalStr.value

                            self.scatter_04_ep[i].x = df.loc[self.flag_filter & self.str_filter,'{}'.format(self.dropdown_xAxis_04_01.value)].to_list()
                            self.scatter_04_ep[i].y = df.loc[self.flag_filter & self.str_filter,list(self.selectMultiple_04_01.value)].sum(axis=1, min_count=1).to_list()

                    if self.selectionSlider_flag.value in [0,1,2]:
                        with self.scatter_04_ep[i].hold_sync():

                            self.flag_filter = df[qc_list].isin([self.selectionSlider_flag.value]).sum(axis=1)==len(self.selectMultiple_04_01.value)
                            self.str_filter = df['H2O_sig_strgth_mean'] > self.floatSlider_signalStr.value

                            self.scatter_04_ep[i].x = df.loc[self.flag_filter & self.str_filter,'{}'.format(self.dropdown_xAxis_04_01.value)].to_list()
                            self.scatter_04_ep[i].y = df.loc[self.flag_filter & self.str_filter,list(self.selectMultiple_04_01.value)].sum(axis=1, min_count=1).to_list()




    def _button_addPathMeta(self, *args):
        if self.tabs.selected_index == 0:
            with self.out_00:
                try:
                    self.folder_path = pathlib.Path(self.text_metafile_00_00.value)
                    readme = self.folder_path.rglob('Readme.txt')
                    readme_df = pd.read_csv(list(readme)[0], delimiter=',')
                    temp_list = [row.to_list() for i,row in readme_df[['rotation', 'lowfrequency','highfrequency','wpl','flagging','name']].iterrows()]
                    a = []
                    self.config_name = []
                    for i in temp_list:
                        self.config_name.append(i[5])
                        a.append('Rotation:{} |LF:{} |HF:{} |WPL:{} |Flag:{}'.format(i[0],i[1],i[2],i[3],i[4]))
                    self.selectMultiple_Meta_00_00.options = a
                except:
                    print('erro')
                    # pass

    def _button_plot(self, *args):

        # if (self.tabs.selected_index == 1) or (self.tabs.selected_index == 4):
        #     self.dfs = []
        #     for i in self.dfs_01_01:
        #         self.dfs += i

        if (self.tabs.selected_index == 1) and (self.accordion_01.selected_index == 0):
            with self.out_01:
                self.x_axis_01_01.label = self.dropdown_xAxis_01_01.value
                self.y_axis_01_01.label = self.dropdown_yAxis_01_01.value

                for i,df in enumerate(self.dfs_01_01):
                    with self.scatter_01_01[i].hold_sync():
                        self.scatter_01_01[i].x = df['{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                        self.scatter_01_01[i].y = df['{}'.format(self.dropdown_yAxis_01_01.value)].to_list()


        if (self.tabs.selected_index == 1) and (self.accordion_01.selected_index == 1):
            with self.out_01:
                self.x_axis_01_02.label = self.dropdown_xAxis_01_01.value
                self.y_axis_01_02.label = self.dropdown_yAxis_01_01.value

                for i,df in enumerate(self.dfs_01_01):
                    with self.scatter_01_02[i].hold_sync():
                        if self.selectionSlider_flag.value == 'All':
                            self.scatter_01_02[i].x = df['{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                            self.scatter_01_02[i].y = df['{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                            # if self.selectionSlider_flag.value == 'All':
                            self.scatter_01_02[i].colors = ['blue']

                        if self.selectionSlider_flag.value in [0,1,2]:
                            self.scatter_01_02[i].x = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_flag.value,'{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                            self.scatter_01_02[i].y = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_flag.value,'{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                            if self.selectionSlider_flag.value == 0:
                                self.scatter_01_02[i].colors = ['green']
                            if self.selectionSlider_flag.value == 1:
                                self.scatter_01_02[i].colors = ['orange']
                            if self.selectionSlider_flag.value == 2:
                                self.scatter_01_02[i].colors = ['red']

        if self.tabs.selected_index == 3:
            with self.out_03:
                self.x_axis_03_01.label = self.dropdown_xAxis_03_01.value
                self.y_axis_03_01.label = self.dropdown_yAxis_03_01.value
                with self.scatter_03_01[0].hold_sync():

                    self.scatter_03_01[0].x = self.dfs_concat_02_01['{}'.format(self.dropdown_xAxis_03_01.value)].to_list()
                    self.scatter_03_01[0].y = self.dfs_concat_02_01['{}'.format(self.dropdown_yAxis_03_01.value)].to_list()

        if self.tabs.selected_index == 4:
            with self.out_04:
                self.x_axis_04_01.label = self.dropdown_xAxis_04_01.value +' and '+ self.dropdown_xAxis_04_02.value
                self.y_axis_04_01.label = ' + '.join(self.selectMultiple_04_01.value) + ' and ' + ' + '.join(self.selectMultiple_04_02.value)

                self.scatter_04_lf[0].x = self.dfs_compare[0][self.dropdown_xAxis_04_02.value].to_list()
                self.scatter_04_lf[0].y = self.dfs_compare[0][list(self.selectMultiple_04_02.value)].sum(axis=1).to_list()
                # print(self.scatter_04_lf)
                # print(self.dropdown_xAxis_04_01.value)
                # print(self.dfs_compare[0]['date_time'])
                for i,df in enumerate(self.dfs_compare):
                    self.scatter_04_ep[i].x = df['{}'.format(self.dropdown_xAxis_04_02.value)].to_list()
                    self.scatter_04_ep[i].y = df[list(self.selectMultiple_04_01.value)].sum(axis=1, min_count=1).to_list()
                # print(df[list(self.selectMultiple_04_01.value)].sum(axis=1,min_count=1).to_list())
        self.brushintsel.marks = self.scatter_04_lf
        self.fig_04_01.interaction = self.brushintsel


    def _button_viewLowFreq(self, *args):
        with self.out_02:
            try:
                self.folder_path_lf = pathlib.Path(self.text_lowfreqfile_02_00.value)
                lf_files = self.folder_path_lf.rglob('TOA5*.flux.dat')
                self.dfs_02_01 = []
                for file in lf_files:
                    self.dfs_02_01.append(pd.read_csv(file, skiprows=[0,2,3], parse_dates=['TIMESTAMP'],na_values='NAN', usecols=self.lf_columns_filtered))

                self.dfs_concat_02_01 = pd.concat(self.dfs_02_01)

                self.dropdown_xAxis_03_01.options = self.dfs_concat_02_01.columns.to_list()
                self.dropdown_yAxis_03_01.options = self.dropdown_xAxis_03_01.options

                self.dropdown_xAxis_04_02.options = self.dropdown_xAxis_03_01.options
                self.selectMultiple_04_02.options = self.dropdown_xAxis_03_01.options

                self.selectMultiple_04_02.observe(self._selectMultiple_compare, 'value')

                self.scatter_03_01 = [bq.Scatter(scales={'x':self.x_scale_03_01,'y':self.y_scale_03_01})]
                self.fig_03_01.marks = self.scatter_03_01

                # print(len(self.dfs_02_01))
                # print(self.dfs_concat_02_01['TIMESTAMP'].max(), self.dfs_concat_02_01['TIMESTAMP'].min())
                # print(self.dfs_concat_02_01.dtypes)

                self.html_02_01.value = "<table> <tr><td><span style='font-weight:bold'>Number of Files:</spam></td> <td>{}</td></tr><tr><td><span style='font-weight:bold'>Begin:</span></td> <td>{}</td></tr> <tr> <td><span style='font-weight:bold'>End:</span></td><td>{}</td>  </tr>".format(len(self.dfs_02_01), self.dfs_concat_02_01['TIMESTAMP'].min(),self.dfs_concat_02_01['TIMESTAMP'].max())

                # print(self.dfs_concat_02_01)
            except:
                pass

    def _selectMultiple_compare(self, *args):
        with self.out_04:
            # print('teste')
            self.scatter_04_lf = [bq.Scatter(scales={'x':self.x_scale_04_01,'y':self.y_scale_04_01}, colors=['red'])]
            self.scatter_04_ep = [bq.Scatter(scales={'x':self.x_scale_04_01, 'y':self.y_scale_04_01}, colors=['blue']) for i in range(len(self.dfs_01_01))]

            self.fig_04_01.marks = self.scatter_04_lf + self.scatter_04_ep
            # print(self.dfs_01_01[0]['date_time'].dtypes)
            # print(self.dfs_concat_02_01['TIMESTAMP'].dtypes)
            self.dfs_compare = [pd.merge(self.dfs_concat_02_01, i, how='outer', on='TIMESTAMP', suffixes=("_lf", "_ep")) for i in self.dfs_01_01]
            # print(self.dfs_compare[0].columns.to_list())
            # print(self.dfs_compare[0][['TIMESTAMP','date_time']]).min()
            self.scatter_04_compare = [bq.Scatter(scales={'x':self.x_scale_04_02,'y':self.y_scale_04_02}, default_opacities=[0.4])]
            self.label_corr = [bq.Label(scales={'x':self.x_scale_04_02, 'y':self.y_scale_04_02},colors=['black'], default_size=20)]

            self.lines_diagonal_04 = [bq.Lines(scales={'x':self.x_scale_04_02, 'y':self.y_scale_04_02}, colors=['black'],line_style='dashed')]

            self.fig_04_02.marks = self.scatter_04_compare + self.label_corr + self.lines_diagonal_04


    def _update(self, *args):
        with self.out_04:
            # print(self.brushintsel.selected)
            # print(self.dfs_compare[0].loc[(self.dfs_compare[0]['TIMESTAMP']>self.brushintsel.selected[0])&(self.dfs_compare[0]['TIMESTAMP']<self.brushintsel.selected[1]),list(self.selectMultiple_04_01.value)].sum(axis=1,min_count=1).to_list())

            self.x_axis_04_02.label = ' + '.join(self.selectMultiple_04_01.value)
            self.y_axis_04_02.label = ' + '.join(self.selectMultiple_04_02.value)

            if self.selectionSlider_flag.value == 'All':
                time_filter_0 = (self.dfs_compare[0]['TIMESTAMP']>self.brushintsel.selected[0])
                time_filter_1 = (self.dfs_compare[0]['TIMESTAMP']<self.brushintsel.selected[1])
                self.scatter_04_compare[0].x = self.dfs_compare[0].loc[time_filter_0 & time_filter_1,list(self.selectMultiple_04_01.value)].sum(axis=1,min_count=1).to_list()
                self.scatter_04_compare[0].y = self.dfs_compare[0].loc[time_filter_0 & time_filter_1,list(self.selectMultiple_04_02.value)].sum(axis=1,min_count=1).to_list()
                df2 = pd.DataFrame()
                df2['EP'] = self.dfs_compare[0].loc[time_filter_0 & time_filter_1 & self.str_filter,list(self.selectMultiple_04_01.value)].sum(axis=1,min_count=1)
                df2['LF'] = self.dfs_compare[0].loc[time_filter_0 & time_filter_1 & self.str_filter,list(self.selectMultiple_04_02.value)].sum(axis=1,min_count=1)

                self.label_corr[0].x = [np.nanmin(self.scatter_04_compare[0].x)]
                self.label_corr[0].y = [np.nanmax(self.scatter_04_compare[0].y)]
                self.label_corr[0].text = ['Pearson: {:.4f}'.format(df2.corr(method='pearson')['LF'][0])]

                self.lines_diagonal_04[0].x = [np.nanmin(self.scatter_04_compare[0].x), np.nanmax(self.scatter_04_compare[0].y)]
                self.lines_diagonal_04[0].y = [np.nanmin(self.scatter_04_compare[0].x), np.nanmax(self.scatter_04_compare[0].y)]
                # print(np.nanmin(self.scatter_04_compare[0].x))
                # print(df2.corr())


            if self.selectionSlider_flag.value in [0,1,2]:
                time_filter_0 = (self.dfs_compare[0]['TIMESTAMP']>self.brushintsel.selected[0])
                time_filter_1 = (self.dfs_compare[0]['TIMESTAMP']<self.brushintsel.selected[1])
                self.scatter_04_compare[0].x = self.dfs_compare[0].loc[time_filter_0 & time_filter_1 & self.flag_filter,list(self.selectMultiple_04_01.value)].sum(axis=1,min_count=1).to_list()
                self.scatter_04_compare[0].y = self.dfs_compare[0].loc[time_filter_0 & time_filter_1 & self.flag_filter ,list(self.selectMultiple_04_02.value)].sum(axis=1,min_count=1).to_list()
                df2 = pd.DataFrame()
                df2['EP'] = self.dfs_compare[0].loc[time_filter_0 & time_filter_1 & self.flag_filter & self.str_filter,list(self.selectMultiple_04_01.value)].sum(axis=1,min_count=1)
                df2['LF'] = self.dfs_compare[0].loc[time_filter_0 & time_filter_1 & self.flag_filter & self.str_filter,list(self.selectMultiple_04_02.value)].sum(axis=1,min_count=1)
                # print(df2.corr()['LF'][0])
                self.label_corr[0].x = [np.nanmin(self.scatter_04_compare[0].x)]
                self.label_corr[0].y = [np.nanmax(self.scatter_04_compare[0].y)]
                self.label_corr[0].text = ['Pearson: {:.4f}'.format(df2.corr(method='pearson')['LF'][0])]

                self.lines_diagonal_04[0].x = [np.nanmin(self.scatter_04_compare[0].x), np.nanmax(self.scatter_04_compare[0].y)]
                self.lines_diagonal_04[0].y = [np.nanmin(self.scatter_04_compare[0].x), np.nanmax(self.scatter_04_compare[0].y)]
            # print(self.scatter_04_compare[0].x)
            # print(self.scatter_04_compare[0].y)
            # print(np.corrcoef(x=self.scatter_04_compare[0].x,y=self.scatter_04_compare[0].y))
