import ipywidgets
import pandas as pd
import numpy as np
import pathlib
# import matplotlib.pyplot as plt
from bqplot import pyplot as plt
import bqplot as bq

class Viewer:
    def __init__(self):

        ####    TAB1    ####
        self.folder_path = ipywidgets.Text(placeholder='Insert Folder Path HERE!')

        self.button01 = ipywidgets.Button(description='Show CSV Files')
        self.button01.on_click(self.button_show_path)

        self.list_files_path = []
        self.select_files = ipywidgets.SelectMultiple(options=self.list_files_path, description='CSVs files',layout={'width': 'max-content'})

        self.tab1 = ipywidgets.VBox([self.folder_path, self.button01, self.select_files])
        ####################

        ####    TAB2    ####
        self.column_names = []
        self.column_x_axis = ipywidgets.Dropdown(options=self.column_names,  description='X-Axis')
        self.column_y_axis = ipywidgets.Dropdown(options=self.column_names,  description='Y-Axis')
        self.button02 = ipywidgets.Button(description='Plot')
        self.button02.on_click(self.button_plot)

        self.x_scale = bq.LinearScale()
        # self.x_scale = bq.DateScale()
        self.y_scale = bq.LinearScale()
        self.xax = bq.Axis(scale=self.x_scale, label='x')
        self.yax = bq.Axis(scale=self.y_scale, label='y', orientation='vertical')

        self.fig = bq.Figure(marks=[], axes=[self.xax, self.yax], animation_duration=1000)

        self.select_files.observe(self._create_lines, 'value')

        self.out = ipywidgets.Output(layout={'border': '1px solid black'})

        self.tab2 = ipywidgets.VBox([ipywidgets.HBox([self.column_x_axis, self.column_y_axis]), self.button02,self.fig,self.out])
        ####################

        ####    TAB3    ####
        self.folder_path_lfdl = ipywidgets.Text(placeholder='Insert Folder Path Low-Frequency Datalogger')
        self.button03 = ipywidgets.Button(description='Show LF CSVs files')
        self.button03.on_click(self._button_show_dl)

        self.list_files_path_lfdl = []
        self.select_files_lfdl = ipywidgets.SelectMultiple(options=self.list_files_path_lfdl, description='Low Frequency DL files', layout={'width': 'max-content'})

        self.tab3 = ipywidgets.VBox([self.folder_path_lfdl, self.button03, self.select_files_lfdl])
        ####################

        ####    TAB4    ####
        self.column_names_lfdl = []
        self.column_x_axis_lfdl = ipywidgets.Dropdown(options=self.column_names_lfdl, description='X-Axis')
        self.column_y_axis_lfdl = ipywidgets.Dropdown(options=self.column_names_lfdl, description='Y-Axis')

        self.button04 = ipywidgets.Button(description='Plot')
        self.button04.on_click(self.button_plot_lfdl)

        self.x_scale02 = bq.LinearScale()
        self.y_scale02 = bq.LinearScale()
        self.xax02 = bq.Axis(scale=self.x_scale02, label='x')
        self.yax02 = bq.Axis(scale=self.y_scale02, label='y', orientation='vertical')

        self.fig02 = bq.Figure(marks=[], axes=[self.xax02, self.yax02], animation_duration=1000)

        self.select_files_lfdl.observe(self._create_lines02, 'value')


        self.out02 = ipywidgets.Output(layout={'border': '1px solid black'})

        self.tab4 = ipywidgets.VBox([ipywidgets.HBox([self.column_x_axis_lfdl, self.column_y_axis_lfdl]), self.button04, self.fig02,self.out02])

        ####################


        self.tabs = ipywidgets.Tab(children=[self.tab1, self.tab2, self.tab3, self.tab4])
        self.tabs.set_title(0, 'Input Data Eddy Covariance')
        self.tabs.set_title(1, 'Plot EddyCovariance data')
        self.tabs.set_title(2, 'Input Data Datalogger LF')
        self.tabs.set_title(3, 'Plot Low Frequency Datalogger')
        display(self.tabs)

    def _create_lines(self, *args):
        self.lines = [bq.Lines(x=[], y=[], scales={'x':self.x_scale, 'y':self.y_scale}) for i in range(len(self.select_files.value))]
        self.fig.marks = self.lines

    def _create_lines02(self, *args):
        self.lines02 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale02, 'y': self.y_scale02}) for i in range(len(self.select_files_lfdl.value))]
        self.fig02.marks = self.lines02

    def button_show_path(self, *args):
        try:
            folder_files_path = pathlib.Path(self.folder_path.value)
            files_path = folder_files_path.rglob('*full_output*.csv')
            self.list_files_path = [i for i in files_path]
            self.select_files.options = self.list_files_path
            self.column_x_axis.options = pd.read_csv(self.list_files_path[0], skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]).columns.to_list()
            self.column_y_axis.options = self.column_x_axis.options
        except:
            # with self.out:
            #     print('fail')
            pass

    def _button_show_dl(self, *args):
        try:
            folder_files_path = pathlib.Path(self.folder_path_lfdl.value)
            files_path = folder_files_path.rglob('TOA5*.flux.dat')
            self.list_files_path_lfdl = [i for i in files_path]
            self.select_files_lfdl.options = self.list_files_path_lfdl
            self.column_x_axis_lfdl.options = pd.read_csv(self.list_files_path_lfdl[0], skiprows=[0,2,3], na_values='NAN', parse_dates=['TIMESTAMP']).columns.to_list()
            self.column_y_axis_lfdl.options = self.column_x_axis_lfdl.options
        except:
            pass


    def button_plot(self, *args):
        # self.out.clear_output()
        # with self.out:
        #     for i in self.select_files.value:
        #         print(i)
        with self.out:

            dataframes = [pd.read_csv(i, skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]) for i in self.select_files.value]

            self.xax.label = self.column_x_axis.value
            self.yax.label = self.column_y_axis.value
            for i, f in enumerate(zip(dataframes, self.select_files.value)):

                self.lines[i].x = f[0]['{}'.format(self.column_x_axis.value)].to_list()
                self.lines[i].y = f[0]['{}'.format(self.column_y_axis.value)].to_list()

    def button_plot_lfdl(self, *args):
        with self.out02:
            dataframes = [pd.read_csv(i, skiprows=[0,2,3], na_values='NAN', parse_dates=['TIMESTAMP']) for i in self.select_files_lfdl.value]
            # print(dataframes[0].head())
            self.xax02.label = self.column_x_axis_lfdl.value
            self.yax02.label = self.column_y_axis_lfdl.value

            for i, f in enumerate(zip(dataframes, self.select_files_lfdl.value)):
                self.lines02[i].x = f[0]['{}'.format(self.column_x_axis_lfdl.value)].to_list()
                self.lines02[i].y = f[0]['{}'.format(self.column_y_axis_lfdl.value)].to_list()
