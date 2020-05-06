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
        self.tabs = ipywidgets.Tab([self.tab00(), self.tab01(), self.tab02(), self.tab03()])


        self.tabs.set_title(0, 'EP - Master Folder')
        self.tabs.set_title(1, 'EP - Simple View')
        self.tabs.set_title(2, 'LowFreq - Master Folder')
        self.tabs.set_title(3, 'LowFreq - Simple View')
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

            # self.panZoom = bq.PanZoom(scales={'x':[self.x_scale_01_01], 'y':[self.y_scale_01_01]})

            self.fig_01_01 = bq.Figure(axes=[self.x_axis_01_01, self.y_axis_01_01],
                                       animation_duration=self.time_interval_01)





            self.x_scale_01_02 = bq.DateScale()
            self.y_scale_01_02 = bq.LinearScale()

            self.x_axis_01_02 = bq.Axis(scale=self.x_scale_01_02)
            self.y_axis_01_02 = bq.Axis(scale=self.y_scale_01_02, orientation='vertical')

            self.fig_01_02 = bq.Figure(axes=[self.x_axis_01_02, self.y_axis_01_02],
                                       animation_duration=self.time_interval_01)

            self.selectionSlider_01_02 = ipywidgets.SelectionSlider(options=[0,1,2,'All'],
                                                                    value='All',
                                                                    description='Flag')
            self.selectionSlider_01_02.observe(self._selectionSlider_flag, 'value')



            self.accordion_01.children = [ipywidgets.VBox([self.fig_01_01]),
                                          ipywidgets.VBox([self.selectionSlider_01_02,
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
            self.y_axis_03_01 = bq.Axis(scale=self.y_scale_03_01, orientation='vertical',label='tes')

            self.fig_03_01 = bq.Figure(axes=[self.x_axis_03_01, self.y_axis_03_01],
                                       animation_duration=self.time_interval_01)

        return ipywidgets.VBox([ipywidgets.HBox([self.dropdown_xAxis_03_01, self.dropdown_yAxis_03_01]),
                                self.button_plot_03_01,
                                self.fig_03_01,
                                self.out_03])

    def _selectMultiple_config(self, *args):
        with self.out_00:
            # print('asd')
            self.scatter_01_01 = []
            self.scatter_01_02 = []

            self.dfs_01_01 = []
            for i in self.selectMultiple_Meta_00_00.index:
                dfs_single_config = []
                full_output_files = self.folder_path.rglob('*{}*_full_output*.csv'.format(self.config_name[i]))
                for file in full_output_files:
                    dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'date_time':['date', 'time']}))

                    self.scatter_01_01.append(bq.Scatter(scales={'x':self.x_scale_01_01, 'y':self.y_scale_01_01}))
                    self.scatter_01_02.append(bq.Scatter(scales={'x':self.x_scale_01_02, 'y':self.y_scale_01_02}))

                self.dfs_01_01.append(dfs_single_config)

                self.dropdown_xAxis_01_01.options = pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]).columns.to_list()
                self.dropdown_yAxis_01_01.options = self.dropdown_xAxis_01_01.options

            self.fig_01_01.marks = self.scatter_01_01
            self.fig_01_02.marks = self.scatter_01_02

    def _selectionSlider_flag(self, *args):
        with self.out_01:
            for i,df in enumerate(self.dfs):
                with self.scatter_01_02[i].hold_sync():
                    if self.selectionSlider_01_02.value == 'All':
                        self.scatter_01_02[i].x = df['{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                        self.scatter_01_02[i].y = df['{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                        self.scatter_01_02[i].colors = ['blue']

                    if self.selectionSlider_01_02.value in [0,1,2]:
                        self.scatter_01_02[i].x = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_01_02.value,'{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                        self.scatter_01_02[i].y = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_01_02.value,'{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                        if self.selectionSlider_01_02.value == 0:
                            self.scatter_01_02[i].colors = ['green']
                        if self.selectionSlider_01_02.value == 1:
                            self.scatter_01_02[i].colors = ['orange']
                        if self.selectionSlider_01_02.value == 2:
                            self.scatter_01_02[i].colors = ['red']

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

        if (self.tabs.selected_index == 1):
            self.dfs = []
            for i in self.dfs_01_01:
                self.dfs += i

        if (self.tabs.selected_index == 1) and (self.accordion_01.selected_index == 0):
            with self.out_01:
                self.x_axis_01_01.label = self.dropdown_xAxis_01_01.value
                self.y_axis_01_01.label = self.dropdown_yAxis_01_01.value

                for i,df in enumerate(self.dfs):
                    with self.scatter_01_01[i].hold_sync():
                        self.scatter_01_01[i].x = df['{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                        self.scatter_01_01[i].y = df['{}'.format(self.dropdown_yAxis_01_01.value)].to_list()


        if (self.tabs.selected_index == 1) and (self.accordion_01.selected_index == 1):
            with self.out_01:
                self.x_axis_01_02.label = self.dropdown_xAxis_01_01.value
                self.y_axis_01_02.label = self.dropdown_yAxis_01_01.value

                for i,df in enumerate(self.dfs):
                    with self.scatter_01_02[i].hold_sync():
                        if self.selectionSlider_01_02.value == 'All':
                            self.scatter_01_02[i].x = df['{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                            self.scatter_01_02[i].y = df['{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                            # if self.selectionSlider_01_02.value == 'All':
                            self.scatter_01_02[i].colors = ['blue']

                        if self.selectionSlider_01_02.value in [0,1,2]:
                            self.scatter_01_02[i].x = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_01_02.value,'{}'.format(self.dropdown_xAxis_01_01.value)].to_list()
                            self.scatter_01_02[i].y = df.loc[df['qc_{}'.format(self.dropdown_yAxis_01_01.value)]==self.selectionSlider_01_02.value,'{}'.format(self.dropdown_yAxis_01_01.value)].to_list()

                            if self.selectionSlider_01_02.value == 0:
                                self.scatter_01_02[i].colors = ['green']
                            if self.selectionSlider_01_02.value == 1:
                                self.scatter_01_02[i].colors = ['orange']
                            if self.selectionSlider_01_02.value == 2:
                                self.scatter_01_02[i].colors = ['red']

        if self.tabs.selected_index == 3:
            with self.out_03:
                self.x_axis_03_01.label = self.dropdown_xAxis_03_01.value
                self.y_axis_03_01.label = self.dropdown_yAxis_03_01.value
                with self.scatter_03_01[0].hold_sync():
                    self.scatter_03_01[0].x = self.dfs_concat_02_01['{}'.format(self.dropdown_xAxis_03_01.value)].to_list()
                    self.scatter_03_01[0].y = self.dfs_concat_02_01['{}'.format(self.dropdown_yAxis_03_01.value)].to_list()


    def _button_viewLowFreq(self, *args):
        with self.out_02:
            try:
                self.folder_path_lf = pathlib.Path(self.text_lowfreqfile_02_00.value)
                lf_files = self.folder_path_lf.rglob('TOA5*.flux.dat')
                self.dfs_02_01 = []
                for file in lf_files:
                    self.dfs_02_01.append(pd.read_csv(file, skiprows=[0,2,3], parse_dates=['TIMESTAMP']))

                self.dfs_concat_02_01 = pd.concat(self.dfs_02_01)

                self.dropdown_xAxis_03_01.options = self.dfs_concat_02_01.columns.to_list()
                self.dropdown_yAxis_03_01.options = self.dropdown_xAxis_03_01.options

                self.scatter_03_01 = [bq.Scatter(scales={'x':self.x_scale_03_01,'y':self.y_scale_03_01})]
                self.fig_03_01.marks = self.scatter_03_01

                # print(len(self.dfs_02_01))
                # print(self.dfs_concat_02_01['TIMESTAMP'].max(), self.dfs_concat_02_01['TIMESTAMP'].min())
                # print(self.dfs_concat_02_01.dtypes)

                self.html_02_01.value = "<table> <tr><td><span style='font-weight:bold'>Number of Files:</spam></td> <td>{}</td></tr><tr><td><span style='font-weight:bold'>Begin:</span></td> <td>{}</td></tr> <tr> <td><span style='font-weight:bold'>End:</span></td><td>{}</td>  </tr>".format(len(self.dfs_02_01), self.dfs_concat_02_01['TIMESTAMP'].min(),self.dfs_concat_02_01['TIMESTAMP'].max())

                # print(self.dfs_concat_02_01)
            except:
                pass
