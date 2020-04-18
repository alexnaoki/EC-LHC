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
        self.time_interval_01 = 500

        self.tabs = ipywidgets.Tab([self.tab01(), self.tab02()])

        self.tabs.set_title(0, 'EP - SimpleView')
        self.tabs.set_title(1, 'EP - Coespectra')

        display(self.tabs)

    def tab01(self):
        """ TAB01
                -General Widgets
                -ACCORDION01
                -ACCORDION02
                -ACCORDION03
        """
        self.folder_path = ipywidgets.Text(placeholder='Insert Folder Path HERE!',
                                           layout=ipywidgets.Layout(width='90%'))

        self.button_showPath_01 = ipywidgets.Button(description='Show CSV Files')
        self.button_showPath_01.on_click(self._button_showPath01)

        self.list_filesPath_01 = []
        self.select_files_01 = ipywidgets.SelectMultiple(options=self.list_filesPath_01,
                                                         description='CSVs Files',
                                                         layout=ipywidgets.Layout(width='99%'))

        self.select_files_01.observe(self._select_observe_01, 'value')

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
        """ TAB02
                -General Widgets
                -ACCORDION01
        """

        self.folder_path_02_01 = ipywidgets.Text(placeholder='Folder Path of CoSpectra files',
                                              layout=ipywidgets.Layout(width='80%'))
        self.button_showPath_02_01 = ipywidgets.Button(description='Show CoSpectra Files',
                                                    layout=ipywidgets.Layout(width='19%'),button_style='primary')
        self.button_showPath_02_01.on_click(self._button_showPath02)

        self.radioButton_02_01 = ipywidgets.RadioButtons(options=['Select Files', 'All'],
                                                      description='Choose how do you want:',
                                                      value=None)

        self.radioButton_02_01.observe(self._radioButton_observe_01, 'value')


        self.list_filesPath_02 = []
        self.select_files_02_01 = ipywidgets.SelectMultiple(description='CoSpectra Files',
                                                         options=self.list_filesPath_02,
                                                         layout=ipywidgets.Layout(width='90%'))
        self.select_files_02_01.observe(self._select_observe_02, 'value')

        self.column_names_02 = []
        self.dropdown_columnX_02_01 = ipywidgets.Dropdown(options=self.column_names_02,
                                                       description='X-Axis')
        self.dropdown_columnY_02_01 = ipywidgets.Dropdown(options=self.column_names_02,
                                                       description='Y-Axis')
        self.out_02 = ipywidgets.Output()

        ####### ACCORDION 01 #######
        self.progress_02_01 = ipywidgets.FloatProgress(value=0, min=0, max=1,
                                                       description='Loading Data:',
                                                       layout=ipywidgets.Layout(width='90%'))

        self.button_plot_02 = ipywidgets.Button(description='Plot Points', button_style='primary')
        self.button_plot_02.on_click(self._button_plot02)

        self.button_plot_03 = ipywidgets.Button(description='Plot Mean', button_style='primary')
        self.button_plot_03.on_click(self._button_plot03)

        self.play_01 = ipywidgets.Play(value=0, min=0,max=100, step=1,
                                       interval=self.time_interval_01)
        self.play_01.observe(self._play_02_01, 'value')

        self.intSlider_02_01 = ipywidgets.IntSlider(description='Timeline',
                                                    layout=ipywidgets.Layout(width='50%'))

        ipywidgets.jslink((self.play_01,'value'),(self.intSlider_02_01,'value'))

        self.x_scale_02_01 = bq.LogScale(min=0.0001,max=100)
        self.y_scale_02_01 = bq.LogScale(min=0.0001, max=10)

        self.x_axis_02_01 = bq.Axis(scale=self.x_scale_02_01,
                                    label='x',
                                    grid_lines='solid',
                                    tick_format='0.4f',
                                    tick_rotate=270)
        self.y_axis_02_01 = bq.Axis(scale=self.y_scale_02_01,
                                    label='y',
                                    orientation='vertical',
                                    tick_format='0.4f')

        self.fig_02_01 = bq.Figure(marks=[],
                                   animation_duration=self.time_interval_01,
                                   layout=ipywidgets.Layout(width='80%'))

        self.fig_02_01.axes = [self.x_axis_02_01, self.y_axis_02_01]
        ####### ############ #######

        ####### ACCORDION 01 #######
        self.folder_path_02_02 = ipywidgets.Text(placeholder='Folder 01 to COMPARE',
                                                 layout=ipywidgets.Layout(width='80%'))
        self.folder_path_02_03 = ipywidgets.Text(placeholder='Folder 02 to COMPARE',
                                                 layout=ipywidgets.Layout(width='80%'))

        self.button_showPath_02_02 = ipywidgets.Button(description='Show BOTH',
                                                       layout=ipywidgets.Layout(height='100%', width='20%'))


        ####### ############ #######





        self.accordion_02 = ipywidgets.Accordion(children=[ipywidgets.VBox([ipywidgets.HBox([self.folder_path_02_01, self.button_showPath_02_01]),
                                                                            self.radioButton_02_01,
                                                                            self.select_files_02_01,
                                                                            ipywidgets.HBox([self.dropdown_columnX_02_01, self.dropdown_columnY_02_01]),
                                                                            self.progress_02_01,
                                                                            ipywidgets.HBox([self.button_plot_02, self.button_plot_03]),
                                                                            ipywidgets.HBox([self.play_01, self.intSlider_02_01]),
                                                                            self.fig_02_01]),
                                                           ipywidgets.VBox([ipywidgets.HBox([ipywidgets.VBox([self.folder_path_02_02,
                                                                                                              self.folder_path_02_03], layout=ipywidgets.Layout(width='100%')), self.button_showPath_02_02])])])
        self.accordion_02.set_title(0, 'Simple Plot')
        self.accordion_02.set_title(1, 'Compare Plot')
        self.accordion_02.selected_index = None

        return ipywidgets.VBox([self.accordion_02,
                                self.out_02])

    def _play_02_01(self, *args):
        """ TAB02
                -ACCORDION01
        Description: Play (Animation) for Scatter given column x and y.
        """

        with self.out_02:
            if self.radioButton_02_01.value == 'Select Files':
                self.fig_02_01.title = self.select_files_02_01.value[self.play_01.value-1].name
                self.scatter_02_01[0].x = self.dfs_02[self.play_01.value-1]['{}'.format(self.dropdown_columnX_02_01.value)].to_list()
                self.scatter_02_01[0].y = self.dfs_02[self.play_01.value-1]['{}'.format(self.dropdown_columnY_02_01.value)].to_list()

            if self.radioButton_02_01.value == 'All':

                self.fig_02_01.title = self.select_files_02_01.options[self.play_01.value-1].name
                self.scatter_02_01[0].x = self.dfs_02[self.play_01.value-1]['{}'.format(self.dropdown_columnX_02_01.value)].to_list()
                self.scatter_02_01[0].y = self.dfs_02[self.play_01.value-1]['{}'.format(self.dropdown_columnY_02_01.value)].to_list()

    def _button_plot03(self, *args):
        """ TAB02
                -ACCORDION01
        Description: Plot MEAN Line given DataFrames (list of).

        """
        # Plot Mean
        with self.out_02:
            dfs_concat = pd.concat(self.dfs_02)
            self.dfs_mean_x = dfs_concat.groupby(by=dfs_concat.index).mean()['{}'.format(self.dropdown_columnX_02_01.value)].to_list()
            self.dfs_mean = dfs_concat.groupby(by=dfs_concat.index).mean()['{}'.format(self.dropdown_columnY_02_01.value)].to_list()

            self.lines_02_01[0].x = self.dfs_mean_x
            self.lines_02_01[0].y = self.dfs_mean

    def _button_showPath01(self, *args):
        """ TAB01
        Description: Given path, show files path results.

        """

        try:
            folder_filesPath = pathlib.Path(self.folder_path.value)
            files_path = folder_filesPath.rglob('*full_output*.csv')
            self.select_files_01.options = [i for i in files_path]
            self.dropdown_columnX_01.options = pd.read_csv(self.select_files_01.options[0],skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]).columns.to_list()
            self.dropdown_columnY_01.options = self.dropdown_columnX_01.options
        except:
            pass

    def _button_showPath02(self, *args):
        """TAB02
        Description: Given path, show files path results.
        """

        try:
            folder_filesPath = pathlib.Path(self.folder_path_02_01.value)
            files_path = folder_filesPath.rglob('*binned_cospectra*.csv')
            self.select_files_02_01.options = [i for i in files_path]
            self.dropdown_columnX_02_01.options = pd.read_csv(self.select_files_02_01.options[0],skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999).columns.to_list()
            self.dropdown_columnY_02_01.options = self.dropdown_columnX_02_01.options
        except:
            pass

    def _select_observe_01(self, *args):
        """ TAB01
                -ACCORDION01
                -ACCORDION02
                -ACCORDION03
        Description: Create Scatter, Lines, Tooltips for figures and define marks.
        """

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
        self.bar_01 = [bq.Bars(scales={'x':self.x_scale_01_04, 'y':bq.LinearScale()},selected_style={'stroke': 'orange', 'fill': 'red'}) for i in range(len(self.select_files_01.value))]
        self.fig_01_04.marks = self.bar_01

        self.tt_scatter_01 = [bq.Scatter(scales={'x':bq.DateScale(), 'y':bq.LinearScale()}) for i in range(len(self.select_files_01.value))]
        self.fig_01_tt.marks = self.tt_scatter_01

        self.bar_01[0].tooltip = self.fig_01_tt
        self.bar_01[0].interactions = {'click':'select','hover':'tooltip'}
        # with self.out_01:
        #     print(self.bar_01[0].selected)

        self.bar_01[0].on_element_click(self._onElementClick_01_01)

    def _select_observe_02(self, *args):
        """ TAB02
                -ACCORDION01
        Description: Create Scatter and read_csv files for select_files_02_01.value.
        """
        with self.out_02:
            self.progress_02_01.value = 0
            self.progress_02_01.bar_style = 'info'
            self.button_plot_02.disabled = True
            self.button_plot_03.disabled = True
            self.play_01.disabled = True
            self.intSlider_02_01.disabled = True
            if self.radioButton_02_01.value == 'Select Files':
                self.scatter_02_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_02_01, 'y':self.y_scale_02_01}) for i in range(len(self.select_files_02_01.value))]
                self.fig_02_01.marks = self.scatter_02_01 + self.lines_02_01

                self.play_01.max = len(self.select_files_02_01.value)
                self.intSlider_02_01.max = self.play_01.max

                self.dfs_02 = []
                for i in self.select_files_02_01.value:
                    self.dfs_02.append(pd.read_csv(i, skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999, engine='python'))
                    self.progress_02_01.value += 1/len(self.select_files_02_01.value)

                self.progress_02_01.bar_style = 'success'
                self.button_plot_02.disabled = False
                self.button_plot_03.disabled = False
                self.play_01.disabled = False
                self.intSlider_02_01.disabled = False

    def _radioButton_observe_01(self, *args):
        """ TAB02
        Description: Create Scatter when choose All and read_csv creating list of DataFrame, or
        create scatter for select files.
        """
        with self.out_02:
            # print('teste')
            self.progress_02_01.value = 0
            self.progress_02_01.bar_style = 'info'
            self.button_plot_02.disabled = True
            self.button_plot_03.disabled = True
            self.play_01.disabled = True
            self.intSlider_02_01.disabled = True

            if self.radioButton_02_01.value == 'All':
                self.scatter_02_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_02_01, 'y':self.y_scale_02_01}) for i in range(len(self.select_files_02_01.options))]

                self.play_01.max = len(self.select_files_02_01.options)
                self.intSlider_02_01.max = self.play_01.max

                self.dfs_02 = []
                for i in self.select_files_02_01.options:
                    self.dfs_02.append(pd.read_csv(i, skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999, engine='python'))
                    self.progress_02_01.value += 1/len(self.select_files_02_01.options)

                self.progress_02_01.bar_style = 'success'
                self.button_plot_02.disabled = False
                self.button_plot_03.disabled = False
                self.play_01.disabled = False
                self.intSlider_02_01.disabled = False

            if self.radioButton_02_01.value == 'Select Files':
                self.scatter_02_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_02_01, 'y':self.y_scale_02_01}) for i in range(len(self.select_files_02_01.value))]

            else:
                print('erro')

            self.lines_02_01 = [bq.Lines(scales={'x':self.x_scale_02_01, 'y':self.y_scale_02_01})]
            self.fig_02_01.marks = self.scatter_02_01 + self.lines_02_01

    def _onElementClick_01_01(self, *args):
        """ TAB01
                -ACCORDION03
        Description: Feed x and y scatter files for the Tooltip, when clicked on the bar.
        """
        with self.out_01:
            try:
                index_selected = list(self.bar_01[0].selected)
                self.str_time = [(dt.datetime(2000,1,1)+ dt.timedelta(minutes=int(i)*30)).strftime('%H:%M') for i in index_selected]
                dt_time = [dt.datetime.strptime(i, '%H:%M').time() for i in self.str_time]
                for i, f in enumerate(self.dfs_01):
                    self.tt_scatter_01[i].x = f.loc[(f['date_time'].dt.time.isin(dt_time))&(f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value), '{}'.format(self.dropdown_columnX_01.value)]
                    self.tt_scatter_01[i].y = f.loc[(f['date_time'].dt.time.isin(dt_time))&(f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value), '{}'.format(self.dropdown_columnY_01.value)]

            except:
                print('erro')

    def _button_plot01(self, *args):
        """ TAB01
                -ACCORDION01
        Description: Create list of DataFrame and feed the scatter x and y.
        """
        with self.out_01:
            self.dfs_01 = [pd.read_csv(i, skiprows=[0,2], na_values=-9999, parse_dates={'date_time':['date', 'time']}) for i in self.select_files_01.value]
            self.x_axis_01_01.label = self.dropdown_columnX_01.value
            self.y_axis_01_01.label = self.dropdown_columnY_01.value

            self.tooltip_01.labels = [self.dropdown_columnX_01.value, self.dropdown_columnY_01.value]

            for i, f in enumerate(self.dfs_01):
                self.scatter_01[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_01[i].y = f['{}'.format(self.dropdown_columnY_01.value)].to_list()

    def _button_plot02(self, *args):
        """ TAB02
                -ACCORDION01
        Description: Plot ALL points selected, using the list of Scatter.
        """
        with self.out_02:
            # print(self.radioButton_02_01.value)
            if self.radioButton_02_01.value == 'Select Files':
                self.x_axis_02_01.label= self.dropdown_columnX_02_01.value
                self.y_axis_02_01.label = self.dropdown_columnY_02_01.value

                for i,f in enumerate(self.dfs_02):
                    self.scatter_02_01[i].x = f['{}'.format(self.dropdown_columnX_02_01.value)].to_list()
                    self.scatter_02_01[i].y = f['{}'.format(self.dropdown_columnY_02_01.value)].to_list()

            elif self.radioButton_02_01.value == 'All':

                self.dfs_02 = [pd.read_csv(i, skiprows=[0,1,2,3,4,5,6,7,8,9,10], na_values=-9999, engine='python') for i in self.select_files_02_01.options]

                self.x_axis_02_01.label= self.dropdown_columnX_02_01.value
                self.y_axis_02_01.label = self.dropdown_columnY_02_01.value

                for i,f in enumerate(self.dfs_02):
                    self.scatter_02_01[i].x = f['{}'.format(self.dropdown_columnX_02_01.value)].to_list()
                    self.scatter_02_01[i].y = f['{}'.format(self.dropdown_columnY_02_01.value)].to_list()

    def _button_diff01(self, *args):
        """ TAB01
                -ACCORDION02
        Description: Create Scatter difference between SAME period. It's necessary to use _button_plot02 function.
        """
        with self.out_01:
            self.x_axis_01_02.label = self.dropdown_columnX_01.value
            self.y_axis_01_02.label = self.dropdown_columnY_01.value
            self.y_axis_01_03.label = self.dropdown_columnY_01.value

            self.scatter_02[0].x = self.dfs_01[0]['{}'.format(self.dropdown_columnX_01.value)].to_list()
            self.scatter_02[0].y = self.dfs_01[0]['{}'.format(self.dropdown_columnY_01.value)] - self.dfs_01[1]['{}'.format(self.dropdown_columnY_01.value)]

            self.lines_01_01[0].x = self.dfs_01[0]['{}'.format(self.dropdown_columnX_01.value)].to_list()
            self.lines_01_01[0].y = (self.dfs_01[0]['{}'.format(self.dropdown_columnY_01.value)]-self.dfs_01[1]['{}'.format(self.dropdown_columnY_01.value)]).cumsum()

    def _button_flag01(self, *args):
        """ TAB01
                -ACCORDION03
        Description: Create Scatter filtered by "qc_" and Bars for determine the quantite.
        """
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
        """ TAB01
                -ACCORDION03
        Description: Slider is connected with the flag.
        """
        with self.out_01:
            for i, f in enumerate(self.dfs_01):
                self.scatter_03[i].x = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value,'{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_03[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value, '{}'.format(self.dropdown_columnY_01.value)].to_list()

                self.bar_01[i].x = [k.strftime('%H:%M') for k in f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value].groupby(by=f['date_time'].dt.time)['{}'.format(self.dropdown_columnY_01.value)].count().index.to_list()]
                self.bar_01[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value].groupby(by=f['date_time'].dt.time)['{}'.format(self.dropdown_columnY_01.value)].count().to_list()
