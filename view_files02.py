import ipywidgets
import pandas as pd
import numpy as np
import pathlib
# import matplotlib.pyplot as plt
from bqplot import pyplot as plt
import bqplot as bq

class Viewer:
    def __init__(self):
        self.tabs = ipywidgets.Tab([self.tab01()])

        self.tabs.set_title(0, 'EP - SimpleView')

        display(self.tabs)

    def tab01(self):
        self.folder_path = ipywidgets.Text(placeholder='Insert Folder Path HERE!', layout=ipywidgets.Layout(width='90%'))
        self.button_showPath_01 = ipywidgets.Button(description='Show CSV Files')
        self.button_showPath_01.on_click(self._button_showPath01)

        self.list_filesPath_01 = []
        self.select_files_01 = ipywidgets.SelectMultiple(options=self.list_filesPath_01, description='CSVs Files', layout=ipywidgets.Layout(width='99%'))
        self.select_files_01.observe(self._select_observeLines, 'value')

        self.column_names_01 = []
        self.dropdown_columnX_01 = ipywidgets.Dropdown(options=self.column_names_01, description='X-Axis')
        self.dropdown_columnY_01 = ipywidgets.Dropdown(options=self.column_names_01, description='Y-Axis')


        ####### ACCORDION 01 #######
        self.button_plot_01 = ipywidgets.Button(description='Plot')
        self.button_plot_01.on_click(self._button_plot01)

        self.x_scale_01_01 = bq.DateScale()
        # self.x_scale_01_01 = bq.LinearScale()
        self.y_scale_01_01 = bq.LinearScale()

        self.x_axis_01_01 = bq.Axis(scale=self.x_scale_01_01, label='x')
        self.y_axis_01_01 = bq.Axis(scale=self.y_scale_01_01, label='y', orientation='vertical')

        self.fig_01_01 = bq.Figure(marks=[], axes=[self.x_axis_01_01, self.y_axis_01_01], animation_duration=500)
        ####### ############# #######

        ####### ACCORDION 02 #######
        self.button_diff_01 = ipywidgets.Button(description='View Difference')
        self.button_diff_01.on_click(self._button_diff01)

        self.x_scale_01_02 = bq.DateScale()
        self.y_scale_01_02 = bq.LinearScale()
        self.y_scale_01_03 = bq.LinearScale()
        self.x_axis_01_02 = bq.Axis(scale=self.x_scale_01_02, label='x')
        self.y_axis_01_02 = bq.Axis(scale=self.y_scale_01_02, label='y', orientation='vertical')
        self.y_axis_01_03 = bq.Axis(scale=self.y_scale_01_03, label='y', orientation='vertical', side='right', grid_lines='none')


        self.fig_01_02 = bq.Figure(marks=[], axes=[self.x_axis_01_02, self.y_axis_01_02, self.y_axis_01_03], animation_duration=500)
        ####### ############# #######

        ####### ACCORDION 03 #######
        self.intslider_01_01 = ipywidgets.IntSlider(value=2, min=0, max=2, step=1, description='Flag')
        self.intslider_01_01.observe(self._intslider_observe_01, 'value')

        self.button_flag_01 = ipywidgets.Button(description='Plot')
        self.button_flag_01.on_click(self._button_flag01)

        self.x_scale_01_03 = bq.DateScale()
        self.y_scale_01_04 = bq.LinearScale()
        self.x_axis_01_03 = bq.Axis(scale=self.x_scale_01_03, label='x')
        self.y_axis_01_04 = bq.Axis(scale=self.y_scale_01_04, label='y', orientation='vertical')

        self.fig_01_03 = bq.Figure(marks=[], axes=[self.x_axis_01_03, self.y_axis_01_04], animation_duration=500)

        ####### ############# #######

        self.accordion_01 = ipywidgets.Accordion()
        self.accordion_01.children = [ipywidgets.VBox([self.button_plot_01, self.fig_01_01]), ipywidgets.VBox([self.button_diff_01, self.fig_01_02]), ipywidgets.VBox([self.button_flag_01, self.intslider_01_01, self.fig_01_03])]

        self.accordion_01.set_title(0, 'Simple Plot')
        self.accordion_01.set_title(1, 'Difference Plot')
        self.accordion_01.set_title(2, 'Flag Plot')


        self.out_01 = ipywidgets.Output()

        return ipywidgets.VBox([ipywidgets.HBox([self.folder_path, self.button_showPath_01]), self.select_files_01, ipywidgets.HBox([self.dropdown_columnX_01, self.dropdown_columnY_01]),self.accordion_01,self.out_01])

    def tab02(self):
        pass


    def _button_showPath01(self, *args):
        try:
            folder_filesPath = pathlib.Path(self.folder_path.value)
            files_path = folder_filesPath.rglob('*full_output*.csv')
            self.select_files_01.options = [i for i in files_path]
            self.dropdown_columnX_01.options = pd.read_csv(self.select_files_01.options[0],skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]).columns.to_list()
            self.dropdown_columnY_01.options = self.dropdown_columnX_01.options
        except:
            pass

    def _select_observeLines(self, *args):
        # self.lines_01 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale_01_01, 'y':self.y_scale_01}) for i in range(len(self.select_files_01.value))]
        # self.fig_01_01.marks = self.lines_01
        # def_tt = bq.Tooltip(fields=['x','y'], formats=['.1f', '3.0f'], labels=['Time', 'Tl-208'])
        # def_tt = ipywidgets.Dropdown(options=['sdfasdf','dfasdfsf','s'])

        self.tooltip_01 = bq.Tooltip(fields=['x','y'], labels=[])
        self.scatter_01 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_01_01, 'y':self.y_scale_01_01}, tooltip=self.tooltip_01) for i in range(len(self.select_files_01.value))]
        self.fig_01_01.marks = self.scatter_01

        self.scatter_02 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_01_02, 'y':self.y_scale_01_02}) for i in range(len(self.select_files_01.value))]
        self.fig_01_02.marks = self.scatter_02
        self.lines_01_01 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale_01_01, 'y':self.y_scale_01_01}) for i in range(len(self.select_files_01.value))]
        self.fig_01_02.marks = self.scatter_02 + self.lines_01_01

        self.scatter_03 = [bq.Scatter(x=[], y=[], scales={'x':self.x_scale_01_03, 'y': self.y_scale_01_04}) for i in range(len(self.select_files_01.value))]
        self.fig_01_03.marks = self.scatter_03

    def _button_plot01(self, *args):
        with self.out_01:
            self.dfs_01 = [pd.read_csv(i, skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]) for i in self.select_files_01.value]

            self.x_axis_01_01.label = self.dropdown_columnX_01.value
            self.y_axis_01_01.label = self.dropdown_columnY_01.value

            self.tooltip_01.labels = [self.dropdown_columnX_01.value, self.dropdown_columnY_01.value]

            for i, f in enumerate(self.dfs_01):
                # self.lines_01[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                # self.lines_01[i].y = f['{}'.format(self.dropdown_columnY_01.value)].to_list()
                # self.scatter_01[i].to
                self.scatter_01[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_01[i].y = f['{}'.format(self.dropdown_columnY_01.value)].to_list()

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

            for i, f in enumerate(self.dfs_01):
                self.scatter_03[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_03[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value, '{}'.format(self.dropdown_columnY_01.value)].to_list()

    def _intslider_observe_01(self, *args):
        with self.out_01:
            for i, f in enumerate(self.dfs_01):
                self.scatter_03[i].x = f['{}'.format(self.dropdown_columnX_01.value)].to_list()
                self.scatter_03[i].y = f.loc[f['qc_{}'.format(self.dropdown_columnY_01.value)]==self.intslider_01_01.value, '{}'.format(self.dropdown_columnY_01.value)].to_list()
