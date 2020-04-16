import ipywidgets
import pandas as pd
import numpy as np
import pathlib
import datetime as dt
# import matplotlib.pyplot as plt
from bqplot import pyplot as plt
import bqplot as bq

class Viewer:
    def __init__(self):
        self.tabs = ipywidgets.Tab([self.tab01(), self.tab02()])

        self.tabs.set_title(0, 'EP - SimpleView')
        self.tabs.set_title(1, 'EP - Coespectra')

        display(self.tabs)

    def tab01(self):
        self.folder_path = ipywidgets.Text(placeholder='Insert Folder Path HERE!',
                                           layout=ipywidgets.Layout(width='90%'))

        self.button_showPath_01 = ipywidgets.Button(description='Show CSV Files')
        self.button_showPath_01.on_click(self._button_showPath01)

        self.list_filesPath_01 = []
        self.select_files_01 = ipywidgets.SelectMultiple(options=self.list_filesPath_01,
                                                         description='CSVs Files',
                                                         layout=ipywidgets.Layout(width='99%'))

        self.select_files_01.observe(self._select_observeLines, 'value')

        self.column_names_01 = []
        self.dropdown_columnX_01 = ipywidgets.Dropdown(options=self.column_names_01,
                                                       description='X-Axis')

        self.dropdown_columnY_01 = ipywidgets.Dropdown(options=self.column_names_01,
                                                       description='Y-Axis')

        ####### ACCORDION 01 #######
        self.button_plot_01 = ipywidgets.Button(description='Plot')
        self.button_plot_01.on_click(self._button_plot01)

        self.x_scale_01_01 = bq.DateScale()
        # self.x_scale_01_01 = bq.LinearScale()
        self.y_scale_01_01 = bq.LinearScale()

        self.x_axis_01_01 = bq.Axis(scale=self.x_scale_01_01, label='x')
        self.y_axis_01_01 = bq.Axis(scale=self.y_scale_01_01, label='y', orientation='vertical')

        self.fig_01_01 = bq.Figure(marks=[],
                                   axes=[self.x_axis_01_01, self.y_axis_01_01],
                                   animation_duration=500)
        ####### ############# #######

        ####### ACCORDION 02 #######
        self.button_diff_01 = ipywidgets.Button(description='View Difference')
        self.button_diff_01.on_click(self._button_diff01)

        self.x_scale_01_02 = bq.DateScale()
        self.y_scale_01_02 = bq.LinearScale()
        self.y_scale_01_03 = bq.LinearScale()
        self.x_axis_01_02 = bq.Axis(scale=self.x_scale_01_02, label='x')
        self.y_axis_01_02 = bq.Axis(scale=self.y_scale_01_02, label='y', orientation='vertical')
        self.y_axis_01_03 = bq.Axis(scale=self.y_scale_01_03, label='y', orientation='vertical', side='right')


        self.fig_01_02 = bq.Figure(marks=[],
                                   axes=[self.x_axis_01_02, self.y_axis_01_02, self.y_axis_01_03],
                                   animation_duration=500)
        ####### ############# #######

        ####### ACCORDION 03 #######
        self.intslider_01_01 = ipywidgets.IntSlider(value=2, min=0, max=2, step=1, description='Flag')
        self.intslider_01_01.observe(self._intslider_observe_01, 'value')

        self.button_flag_01 = ipywidgets.Button(description='Plot')
        self.button_flag_01.on_click(self._button_flag01)


        self.x_scale_01_03 = bq.DateScale()
        # self.x_scale_01_03 = bq.LinearScale()
        self.y_scale_01_04 = bq.LinearScale()
        self.x_axis_01_03 = bq.Axis(scale=self.x_scale_01_03, label='x')
        self.y_axis_01_04 = bq.Axis(scale=self.y_scale_01_04, label='y', orientation='vertical')

        self.fig_01_03 = bq.Figure(marks=[],
                                   axes=[self.x_axis_01_03, self.y_axis_01_04],
                                   animation_duration=500)


        self.x_scale_01_04 = bq.OrdinalScale(domain=[(dt.datetime(2000,1,1) + dt.timedelta(minutes=i*30)).strftime('%H:%M') for i in range(48)])

        self.x_axis_01_04 = bq.Axis(scale=self.x_scale_01_04, label='x', tick_rotate=270)
        #
        # self.y_axis_01_06 = bq.Axis(scale=self.y_scale_01_05, label='y', orientation='vertical')

        self.y_axis_bar = bq.Axis(scale=bq.LinearScale(), label='y', side='right',orientation='vertical', grid_lines='solid')
        # self.y_axis_bar_02 = bq.Axis(scale=bq.LinearScale(),label='y2', orientation='vertical')

        self.fig_01_04 = bq.Figure(marks=[],
                                   axes=[self.x_axis_01_04, self.y_axis_bar],
                                   animation_duration=500)


        self.fig_01_tt = bq.Figure(marks=[],
                                   axes=[self.x_axis_01_03, self.y_axis_01_04],
                                   animation_duration=500)
        # self.brush_01 = bq.BrushIntervalSelector(scale=self.x_axis_01_03)


        ####### ############# #######

        self.accordion_01 = ipywidgets.Accordion()
        self.accordion_01.children = [ipywidgets.VBox([self.button_plot_01, self.fig_01_01]),
                                      ipywidgets.VBox([self.button_diff_01, self.fig_01_02]),
                                      ipywidgets.VBox([self.button_flag_01, self.intslider_01_01,self.fig_01_03,self.fig_01_04])]

        self.accordion_01.set_title(0, 'Simple Plot')
        self.accordion_01.set_title(1, 'Difference Plot')
        self.accordion_01.set_title(2, 'Flag Plot')


        self.out_01 = ipywidgets.Output()

        return ipywidgets.VBox([ipywidgets.HBox([self.folder_path, self.button_showPath_01]),
                                self.select_files_01,
                                ipywidgets.HBox([self.dropdown_columnX_01, self.dropdown_columnY_01]),
                                self.accordion_01,self.out_01])

    def tab02(self):
        self.folder_path_02 = ipywidgets.Text(placeholder='Folder Path of CoSpectra files',
                                              layout=ipywidgets.Layout(width='80%'))
        self.button_showPath_02 = ipywidgets.Button(description='Show CoSpectra Files',
                                                    layout=ipywidgets.Layout(width='19%'))
        self.button_showPath_02.on_click(self._button_showPath02)

        self.radioButton_01 = ipywidgets.RadioButtons(options=['Select Files', 'All'],
                                                      description='Choose how do you want:')

        self.radioButton_01.observe(self._radioButton_observe_01, 'value')


        self.list_filesPath_02 = []
        self.select_files_02 = ipywidgets.SelectMultiple(description='CoSpectra Files',
                                                         options=self.list_filesPath_02,
                                                         layout=ipywidgets.Layout(width='90%'))
        self.select_files_02.observe(self._select_observe_02, 'value')

        self.column_names_02 = []
        self.dropdown_columnX_02 = ipywidgets.Dropdown(options=self.column_names_02,
                                                       description='X-Axis')
        self.dropdown_columnY_02 = ipywidgets.Dropdown(options=self.column_names_02,
                                                       description='Y-Axis')
        self.out_02 = ipywidgets.Output()

        ####### ACCORDION 01 #######
        self.button_plot_02 = ipywidgets.Button(description='Plot')
        self.button_plot_02.on_click(self._button_plot02)

        self.play_01 = ipywidgets.Play(value=0, min=0,max=100, step=1,interval=250)
        self.play_01.observe(self._play_teste, 'value')

        with self.out_02:
            print(self.radioButton_01.value)
            self.x_scale_02_01 = bq.LogScale()
            self.y_scale_02_01 = bq.LogScale()

            self.x_axis_02_01 = bq.Axis(scale=self.x_scale_02_01, label='x', grid_lines='solid')
            self.y_axis_02_01 = bq.Axis(scale=self.y_scale_02_01, label='y', orientation='vertical')
            print(self.x_axis_02_01)

            self.fig_02_01 = bq.Figure(marks=[],
                                       animation_duration=1000,
                                       layout=ipywidgets.Layout(width='80%'))
            self.fig_02_01.axes = [self.x_axis_02_01, self.y_axis_02_01]
            # self.scatter_02_01 = [bq.Lines(x=[0,1,2], y=[5,10,23],scales={'x':bq.LinearScale(), 'y':bq.LinearScale()})]
            # self.fig_02_01.marks = self.scatter_02_01
        ####### ############ #######

        self.accordion_02 = ipywidgets.Accordion(children=[ipywidgets.VBox([self.button_plot_02,
                                                                            self.play_01,
                                                                            self.fig_02_01])])
        self.accordion_02.set_title(0, 'Simple Plot')

        return ipywidgets.VBox([ipywidgets.HBox([self.folder_path_02, self.button_showPath_02]),
                                self.radioButton_01,
                                self.select_files_02,
                                ipywidgets.HBox([self.dropdown_columnX_02, self.dropdown_columnY_02]),
                                self.accordion_02,
                                self.out_02])


    # def _brush_teste01(self, *args):
    #     # self.tt_scatter_01[0].x =
    #     with self.out_01:

    def _play_teste(self, *args):
        with self.out_02:

            # print('teste')
            # print(self.play_01.value)
            if self.radioButton_01.value == 'Select Files':

                self.dfs_02 = [pd.read_csv(i, skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999, engine='python') for i in self.select_files_02.value]
                self.play_01.max = len(self.dfs_02)
                print(self.play_01.value)
                self.fig_02_01.title = self.select_files_02.value[self.play_01.value-1].name

                # for i,f in enumerate(self.dfs_02):
                self.scatter_02_01[0].x = self.dfs_02[self.play_01.value-1]['{}'.format(self.dropdown_columnX_02.value)].to_list()
                self.scatter_02_01[0].y = self.dfs_02[self.play_01.value-1]['{}'.format(self.dropdown_columnY_02.value)].to_list()

    def _button_showPath01(self, *args):
        try:
            folder_filesPath = pathlib.Path(self.folder_path.value)
            files_path = folder_filesPath.rglob('*full_output*.csv')
            self.select_files_01.options = [i for i in files_path]
            self.dropdown_columnX_01.options = pd.read_csv(self.select_files_01.options[0],skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]).columns.to_list()
            self.dropdown_columnY_01.options = self.dropdown_columnX_01.options
        except:
            pass

    def _button_showPath02(self, *args):
        try:
            folder_filesPath = pathlib.Path(self.folder_path_02.value)
            files_path = folder_filesPath.rglob('*binned_cospectra*.csv')
            self.select_files_02.options = [i for i in files_path]
            self.dropdown_columnX_02.options = pd.read_csv(self.select_files_02.options[0],skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999).columns.to_list()
            self.dropdown_columnY_02.options = self.dropdown_columnX_02.options
        except:
            pass

    def _select_observeLines(self, *args):

        self.tooltip_01 = bq.Tooltip(fields=['x','y'], labels=[])
        self.scatter_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_01_01, 'y':self.y_scale_01_01}, tooltip=self.tooltip_01) for i in range(len(self.select_files_01.value))]
        self.fig_01_01.marks = self.scatter_01

        self.scatter_02 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_01_02, 'y':self.y_scale_01_02}) for i in range(len(self.select_files_01.value))]
        self.fig_01_02.marks = self.scatter_02
        self.lines_01_01 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale_01_01, 'y':self.y_scale_01_01}) for i in range(len(self.select_files_01.value))]
        self.fig_01_02.marks = self.scatter_02 + self.lines_01_01

        self.scatter_03 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_01_03, 'y': self.y_scale_01_04}) for i in range(len(self.select_files_01.value))]
        # self.fig_01_03.marks = self.scatter_03

        self.label_01 = [bq.Label(x=[], y=[],text=[],scales={'x':self.x_scale_01_03, 'y':self.y_scale_01_04}) for i in range(len(self.select_files_01.value))]
        self.fig_01_03.marks = self.scatter_03 + self.label_01
# ,selected_style={'stroke': 'orange', 'fill': 'red'}
        self.bar_01 = [bq.Bars(scales={'x':self.x_scale_01_04, 'y':bq.LinearScale()},selected_style={'stroke': 'orange', 'fill': 'red'}) for i in range(len(self.select_files_01.value))]
        self.fig_01_04.marks = self.bar_01

        self.tt_scatter_01 = [bq.Scatter(scales={'x':bq.DateScale(), 'y':bq.LinearScale()}) for i in range(len(self.select_files_01.value))]
        self.fig_01_tt.marks = self.tt_scatter_01

        self.bar_01[0].tooltip = self.fig_01_tt
        self.bar_01[0].interactions = {'click':'select','hover':'tooltip'}
        # with self.out_01:
        #     print(self.bar_01[0].selected)

        self.bar_01[0].on_element_click(self._onelementclick_teste)

    def _select_observe_02(self, *args):
        with self.out_02:

            self.scatter_02_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_02_01, 'y':self.y_scale_02_01}) for i in range(len(self.select_files_02.value))]
            self.fig_02_01.marks = self.scatter_02_01
            # self.scatter_02_01 = [bq.Scatter(x=[0,1,2], y=[5,10,23],scales={'x':bq.LinearScale(), 'y':bq.LinearScale()})]
            # self.fig_02_01.marks = self.scatter_02_01

    def _radioButton_observe_01(self, *args):
        with self.out_02:
            print('teste')
            if self.radioButton_01.value == 'All':
                self.scatter_02_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_02_01, 'y':self.y_scale_02_01}) for i in range(len(self.select_files_02.options))]
                self.fig_02_01.marks = self.scatter_02_01
            else:
                print('erro')
                pass

    def _onelementclick_teste(self, *args):
        with self.out_01:
            try:

                index_selected = list(self.bar_01[0].selected)


                self.str_time = [(dt.datetime(2000,1,1)+ dt.timedelta(minutes=int(i)*30)).strftime('%H:%M') for i in index_selected]
                # print(self.str_time)
                dt_time = [dt.datetime.strptime(i, '%H:%M').time() for i in self.str_time]
                # print(dt_time)
                for i, f in enumerate(self.dfs_01):
                    # self.tt_scatter_01[i].x = f.loc[f['date_time'].dt.time == dt_time, '{}'.format(self.dropdown_columnX_01.value)]
                    # self.tt_scatter_01[i].y = f.loc[f['date_time'].dt.time == dt_time, '{}'.format(self.dropdown_columnY_01.value)]
                    # self.tt_scatter_01[i].x = f.loc[(f['date_time'].dt.time == dt_time)&(f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value), '{}'.format(self.dropdown_columnX_01.value)]
                    # self.tt_scatter_01[i].y = f.loc[(f['date_time'].dt.time == dt_time)&(f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value), '{}'.format(self.dropdown_columnY_01.value)]

                    self.tt_scatter_01[i].x = f.loc[(f['date_time'].dt.time.isin(dt_time))&(f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value), '{}'.format(self.dropdown_columnX_01.value)]
                    self.tt_scatter_01[i].y = f.loc[(f['date_time'].dt.time.isin(dt_time))&(f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value), '{}'.format(self.dropdown_columnY_01.value)]

            except:
                print('erro')



    def _button_plot01(self, *args):
        with self.out_01:
            self.dfs_01 = [pd.read_csv(i, skiprows=[0,2], na_values=-9999, parse_dates={'date_time':['date', 'time']}) for i in self.select_files_01.value]
 # parse_dates=[['date','time']]
            self.x_axis_01_01.label = self.dropdown_columnX_01.value
            self.y_axis_01_01.label = self.dropdown_columnY_01.value

            self.tooltip_01.labels = [self.dropdown_columnX_01.value, self.dropdown_columnY_01.value]

            for i, f in enumerate(self.dfs_01):
                # self.lines_01[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                # self.lines_01[i].y = f['{}'.format(self.dropdown_columnY_01.value)].to_list()
                # self.scatter_01[i].to
                self.scatter_01[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_01[i].y = f['{}'.format(self.dropdown_columnY_01.value)].to_list()

    def _button_plot02(self, *args):
        with self.out_02:
            print(self.radioButton_01.value)
            if self.radioButton_01.value == 'Select Files':
                self.dfs_02 = [pd.read_csv(i, skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999, engine='python') for i in self.select_files_02.value]
                self.x_axis_02_01.label= self.dropdown_columnX_02.value
                self.y_axis_02_01.label = self.dropdown_columnY_02.value

                for i,f in enumerate(self.dfs_02):
                    self.scatter_02_01[i].x = f['{}'.format(self.dropdown_columnX_02.value)].to_list()
                    self.scatter_02_01[i].y = f['{}'.format(self.dropdown_columnY_02.value)].to_list()

            elif self.radioButton_01.value == 'All':

                self.dfs_02 = [pd.read_csv(i, skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999, engine='python') for i in self.select_files_02.options]
                print(len(self.dfs_02))

                self.x_axis_02_01.label= self.dropdown_columnX_02.value
                self.y_axis_02_01.label = self.dropdown_columnY_02.value

                for i,f in enumerate(self.dfs_02):
                    self.scatter_02_01[i].x = f['{}'.format(self.dropdown_columnX_02.value)].to_list()
                    self.scatter_02_01[i].y = f['{}'.format(self.dropdown_columnY_02.value)].to_list()

    def _button_diff01(self, *args):
        with self.out_01:
            self.x_axis_01_02.label = self.dropdown_columnX_01.value
            self.y_axis_01_02.label = self.dropdown_columnY_01.value
            self.y_axis_01_03.label = self.dropdown_columnY_01.value

            self.scatter_02[0].x = self.dfs_01[0]['{}'.format(self.dropdown_columnX_01.value)].to_list()
            self.scatter_02[0].y = self.dfs_01[0]['{}'.format(self.dropdown_columnY_01.value)] - self.dfs_01[1]['{}'.format(self.dropdown_columnY_01.value)]

            self.lines_01_01[0].x = self.dfs_01[0]['{}'.format(self.dropdown_columnX_01.value)].to_list()
            self.lines_01_01[0].y = (self.dfs_01[0]['{}'.format(self.dropdown_columnY_01.value)]-self.dfs_01[1]['{}'.format(self.dropdown_columnY_01.value)]).cumsum()

    def _button_flag01(self, *args):
        with self.out_01:
            self.x_axis_01_03.label = self.dropdown_columnX_01.value
            self.y_axis_01_04.label = self.dropdown_columnY_01.value

            self.x_axis_01_04.label = 'Time'
            self.y_axis_bar.label = 'Count'

            for i, f in enumerate(self.dfs_01):
                self.scatter_03[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_03[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value, '{}'.format(self.dropdown_columnY_01.value)].to_list()

                self.bar_01[i].x = [k.strftime('%H:%M') for k in f.groupby(by=f['date_time'].dt.time)['{}'.format(self.dropdown_columnY_01.value)].count().index.to_list()]
                self.bar_01[i].y = f.groupby(by=f['date_time'].dt.time)['{}'.format(self.dropdown_columnY_01.value)].count().to_list()

    def _intslider_observe_01(self, *args):
        with self.out_01:
            for i, f in enumerate(self.dfs_01):
                self.scatter_03[i].x = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value,'{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_03[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value, '{}'.format(self.dropdown_columnY_01.value)].to_list()

                self.bar_01[i].x = [k.strftime('%H:%M') for k in f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value].groupby(by=f['date_time'].dt.time)['{}'.format(self.dropdown_columnY_01.value)].count().index.to_list()]
                self.bar_01[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value].groupby(by=f['date_time'].dt.time)['{}'.format(self.dropdown_columnY_01.value)].count().to_list()
