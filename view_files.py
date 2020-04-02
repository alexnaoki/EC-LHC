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

        # self.x_scale02 = bq.LinearScale()
        self.x_scale02 = bq.DateScale()
        self.y_scale02 = bq.LinearScale()
        self.xax02 = bq.Axis(scale=self.x_scale02, label='x')
        self.yax02 = bq.Axis(scale=self.y_scale02, label='y', orientation='vertical')

        self.fig02 = bq.Figure(marks=[], axes=[self.xax02, self.yax02], animation_duration=1000)

        self.select_files_lfdl.observe(self._create_lines02, 'value')


        self.out02 = ipywidgets.Output(layout={'border': '1px solid black'})

        self.tab4 = ipywidgets.VBox([ipywidgets.HBox([self.column_x_axis_lfdl, self.column_y_axis_lfdl]), self.button04, self.fig02,self.out02])
        ####################

        ####    TAB5    ####
        self.path_01 = ipywidgets.Text(placeholder='Insert CSV files from EC')
        self.path_02 = ipywidgets.Text(placeholder='Insert LowFreq data')

        self.button05 = ipywidgets.Button(description='Show All')
        self.button05.on_click(self._button_show_all)

        self.list_files01 = []
        self.list_files02 = []
        self.select01 = ipywidgets.SelectMultiple(options=self.list_files01,description='CSVs files',layout={'width': 'max-content'})
        self.select02 = ipywidgets.SelectMultiple(options=self.list_files02,description='LowFreq files',layout={'width': 'max-content'})
        self.select01.observe(self._create_lines03, 'value')
        self.select02.observe(self._create_lines04, 'value')
        # self.select02.observe(self._create_lines03)

        self.column_names_01 = []
        self.column_y_01 = ipywidgets.Dropdown(options=self.column_names_01, description='Y-Axis EC')
        self.column_names_02 = []
        self.column_y_02 = ipywidgets.Dropdown(options=self.column_names_02, description='Y-Axis LowFreq')

        self.button06 = ipywidgets.Button(description='Plot Compare')
        self.button06.on_click(self._button_plot_compare)

        self.x_scale03 = bq.DateScale()
        self.y_scale03 = bq.LinearScale()
        self.xax03 = bq.Axis(scale=self.x_scale03, label='x')
        self.yax03_01 = bq.Axis(scale=self.y_scale03, label='y1', orientation='vertical', side='left')
        self.yax03_02 = bq.Axis(scale=self.y_scale03, label='y2', orientation='vertical', side='right')
        self.fig03 = bq.Figure(marks=[], axes=[self.xax03, self.yax03_01, self.yax03_02], animation_duration=1000)


        self.out03 = ipywidgets.Output(layout={'border':'1px solid black'})

        self.tab5 = ipywidgets.VBox([ipywidgets.HBox([self.path_01, self.path_02]),self.button05,ipywidgets.VBox([self.select01, self.select02]),ipywidgets.HBox([self.column_y_01, self.column_y_02]), self.button06, self.fig03, self.out03])
        ####################

        self.tabs = ipywidgets.Tab(children=[self.tab1, self.tab2, self.tab3, self.tab4, self.tab5])
        self.tabs.set_title(0, 'Input Data Eddy Covariance')
        self.tabs.set_title(1, 'Plot EddyCovariance data')
        self.tabs.set_title(2, 'Input Data Datalogger LF')
        self.tabs.set_title(3, 'Plot Low Frequency Datalogger')
        self.tabs.set_title(4, 'Compare')
        display(self.tabs)

    def _create_lines(self, *args):
        self.lines = [bq.Lines(x=[], y=[], scales={'x':self.x_scale, 'y':self.y_scale}) for i in range(len(self.select_files.value))]
        self.fig.marks = self.lines

    def _create_lines02(self, *args):
        self.lines02 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale02, 'y': self.y_scale02}) for i in range(len(self.select_files_lfdl.value))]
        self.fig02.marks = self.lines02

    def _create_lines03(self, *args):
        with self.out03:
            self.lines03 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale03, 'y': self.y_scale03},colors=['red']) for i in range(len(self.select01.value))]
            # self.fig03.marks = self.lines03
            # self.lines04 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale03, 'y': self.y_scale03}) for i in range(len(self.select02.value))]

            # self.fig03.marks = [self.lines03, self.lines04]
            # print(self.lines03)
            # print(self.lines04)

    def _create_lines04(self, *args):
        with self.out03:
            self.lines04 = [bq.Lines(x=[], y=[], scales={'x':self.x_scale03, 'y': self.y_scale03},colors=['blue']) for i in range(len(self.select02.value))]
            # self.lines03.append(np.squeeze(self.lines04))
            self.lines03 = self.lines03 + self.lines04
            self.fig03.marks = self.lines03

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

    def _button_show_all(self, *args):
        try:
            folder01 = pathlib.Path(self.path_01.value)
            files01 = folder01.rglob('*full_output*.csv')
            self.select01.options = [i for i in files01]
            self.column_y_01.options = pd.read_csv(self.select01.options[0], skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]).columns.to_list()

            folder02 = pathlib.Path(self.path_02.value)
            files02 = folder02.rglob('TOA5*.flux.dat')
            self.select02.options = [i for i in files02]
            self.column_y_02.options = pd.read_csv(self.select02.options[0], skiprows=[0,2,3], na_values='NAN', parse_dates=['TIMESTAMP']).columns.to_list()
        except:
            pass

    def button_plot(self, *args):
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

    def _button_plot_compare(self, *args):
        dataframes01 = [pd.read_csv(i, skiprows=[0,2], na_values=-9999, parse_dates=[['date','time']]) for i in self.select01.value]
        dataframes02 = [pd.read_csv(i, skiprows=[0,2,3], na_values='NAN', parse_dates=['TIMESTAMP']) for i in self.select02.value]

        self.xax03.label = 'Time'
        self.yax03_01.label = self.column_y_01.value
        self.yax03_02.label = self.column_y_02.value

        for i,f in enumerate(dataframes01):
            self.lines03[i].x = f['{}'.format('date_time')].to_list()
            self.lines03[i].y = f['{}'.format(self.column_y_01.value)].to_list()

        for i, f in enumerate(dataframes02, start=len(dataframes01)):
            self.lines03[i].x = f['{}'.format('TIMESTAMP')].to_list()
            self.lines03[i].y = f['{}'.format(self.column_y_02.value)].to_list()
