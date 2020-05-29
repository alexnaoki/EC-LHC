import ipywidgets
import numpy as np
import pandas as pd
import pathlib
import datetime as dt
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, gridplot
from bokeh.models import ColumnDataSource, RangeTool, Circle, Slope, Label
from bokeh.layouts import gridplot, column, row

class view_wind:
    def __init__(self):

        self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
 'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',
  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var']

        self.lf_columns_filtered = ['TIMESTAMP','Hs','u_star','Ts_stdev','Ux_stdev','Uy_stdev','Uz_stdev','Ux_Avg', 'Uy_Avg', 'Uz_Avg',
                                    'Ts_Avg', 'LE_wpl', 'Hc','H2O_mean', 'amb_tmpr_Avg', 'amb_press_mean', 'Tc_mean', 'rho_a_mean','CO2_sig_strgth_mean',
                                    'H2O_sig_strgth_mean','T_tmpr_rh_mean', 'e_tmpr_rh_mean', 'e_sat_tmpr_rh_mean', 'H2O_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                     'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
                                      'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)', 'precip_Tot', 'panel_tmpr_Avg']
        self.TOOLS="pan,wheel_zoom,box_select,lasso_select,save,reset"

        output_notebook()

        self.tabs = ipywidgets.Tab([self.tab00(), self.tab01(), self.tab02()])

        self.tabs.set_title(0, 'EP - Master Folder')
        self.tabs.set_title(1, 'LowFreq - Master Folder')
        self.tabs.set_title(2, 'Plot')

        self.source_ep = ColumnDataSource(data=dict(x=[], u=[], u_c=[], v=[],v_c=[],w=[],w_c=[],teste_x=[],teste_y=[]))
        self.fig_01 = figure(title='Uncorrected', plot_height=250, plot_width=700, x_axis_type='datetime', tools=self.TOOLS)
        circle_u_uncorrected = self.fig_01.circle(x='x',y='u', source=self.source_ep,color='blue', legend_label='u')
        circle_v_uncorrected = self.fig_01.circle(x='x',y='v', source=self.source_ep,color='red', legend_label='v')
        circle_w_uncorrected = self.fig_01.circle(x='x',y='w', source=self.source_ep,color='green', legend_label='w')

        self.fig_01.legend.location = 'top_left'
        self.fig_01.legend.click_policy='hide'

        self.fig_02 = figure(title='Corrected', plot_height=250, plot_width=700, x_axis_type='datetime', x_range=self.fig_01.x_range)
        circle_u_corrected = self.fig_02.circle(x='x',y='u_c', source=self.source_ep,color='blue', legend_label='u_c')
        circle_v_corrected = self.fig_02.circle(x='x',y='v_c', source=self.source_ep,color='red', legend_label='v_c')
        circle_w_corrected = self.fig_02.circle(x='x',y='w_c', source=self.source_ep,color='green', legend_label='w_c')
        self.fig_02.legend.location = 'top_left'
        self.fig_02.legend.click_policy='hide'


        # wind_data = dict(inner=)
        self.source_ep2 = ColumnDataSource(data=dict(inner=[0],outer=[1],start=[0],end=[2]))

        self.fig_03 = figure(title='tes',plot_height=500, plot_width=500)
        self.fig_03.xgrid.grid_line_color = None
        self.fig_03.ygrid.grid_line_color = None
        # self.fig_03.wedge(x=[0,0,0], y=[0,0,0], radius=[1,2,3], start_angle=[0,0.5,1], end_angle=[0.5,1,1.5])
        # bars = self.fig_03.vbar(x='teste_x', width=0.5, bottom=0, top='teste_y', source=self.source_ep)
        wedge = self.fig_03.annular_wedge(x=0, y=0, inner_radius='inner', outer_radius='outer', start_angle='start', end_angle='end',color='#FF00FF',source=self.source_ep2)
        circle = self.fig_03.circle(x=0, y=0, radius=[0.25,0.5,0.75,1], fill_color=None,line_color='white')

        self.fig_03.annular_wedge(x=0, y=0, inner_radius='inner', outer_radius='outer', start_angle='start', end_angle='end', line_color='white',fill_color=None, line_width=1,source=self.source_ep2)

        c = column([self.fig_01, self.fig_02])
        # c = gridplot([[self.fig_01],[self.fig_02]])
        display(self.tabs)
        show(row(c,self.fig_03), notebook_handle=True)

    def tab00(self):
        self.out_00 = ipywidgets.Output()
        with self.out_00:
            self.path_EP = ipywidgets.Text(placeholder='Path EP output',
                                           layout=ipywidgets.Layout(width='90%'))
            self.button_path_ep = ipywidgets.Button(description='Show EP')
            self.button_path_ep.on_click(self._button_Path)

            self.select_meta = ipywidgets.Select(description='Configs:',
                                                                    layout=ipywidgets.Layout(width='90%'),
                                                                    style={'description_width':'initial'})
            self.select_meta.observe(self._select_config, 'value')

        return ipywidgets.VBox([ipywidgets.HBox([self.path_EP, self.button_path_ep]),
                                self.select_meta,
                                self.out_00])

    def tab01(self):
        self.out_01 = ipywidgets.Output()
        with self.out_01:
            self.path_LF = ipywidgets.Text(placeholder='Path LF output',
                                           layout=ipywidgets.Layout(width='90%'))
            self.button_path_lf = ipywidgets.Button(description='Show LF')
            self.button_path_lf.on_click(self._button_Path)

            self.html_lf = ipywidgets.HTML()

        return ipywidgets.VBox([self.out_01,
                                ipywidgets.HBox([self.path_LF, self.button_path_lf]),
                                self.html_lf])
    def tab02(self):
        self.out_02 = ipywidgets.Output()
        with self.out_02:
            self.button_plot = ipywidgets.Button(description='Plot')
            # self.button_plot.on_click(self.update_ep)
            self.button_plot.on_click(self._button_plot)

            self.date_range = ipywidgets.SelectionRangeSlider(description='Date Range:', options=[1,2], layout={'width': '1000px'})
            self.date_range.observe(self.update_byDate, 'value')

            self.hour_range = ipywidgets.SelectionRangeSlider(description='Hour Range:', options=[1,2], layout=ipywidgets.Layout(width='1000px'))
            self.hour_range.observe(self.update_byDate, 'value')

        return ipywidgets.VBox([self.out_02,
                                self.button_plot,
                                self.date_range,
                                self.hour_range])


    def _button_Path(self, *args):
        if self.tabs.selected_index == 0:
            with self.out_00:
                try:
                    self.folder_path_ep = pathlib.Path(self.path_EP.value)
                    readme = self.folder_path_ep.rglob('Readme.txt')
                    readme_df = pd.read_csv(list(readme)[0], delimiter=',')
                    temp_list = [row.to_list() for i,row in readme_df[['rotation', 'lowfrequency','highfrequency','wpl','flagging','name']].iterrows()]
                    a = []
                    self.config_name = []
                    for i in temp_list:
                        self.config_name.append(i[5])
                        a.append('Rotation:{} |LF:{} |HF:{} |WPL:{} |Flag:{}'.format(i[0],i[1],i[2],i[3],i[4]))
                    self.select_meta.options = a
                except:
                    print('Erro')

        if self.tabs.selected_index == 1:
            with self.out_01:
                try:
                    self.folder_path_lf = pathlib.Path(self.path_LF.value)
                    lf_files = self.folder_path_lf.rglob('TOA5*.flux.dat')
                    self.dfs_02_01 = []
                    for file in lf_files:
                        # print(file)
                        self.dfs_02_01.append(pd.read_csv(file, skiprows=[0,2,3], parse_dates=['TIMESTAMP'],na_values='NAN', usecols=self.lf_columns_filtered))

                    self.dfs_concat_02_01 = pd.concat(self.dfs_02_01)

                    # self.dropdown_yAxis_lf.options = self.lf_columns_filtered
                    self.html_lf.value = "<table> <tr><td><span style='font-weight:bold'>Number of Files:</spam></td> <td>{}</td></tr><tr><td><span style='font-weight:bold'>Begin:</span></td> <td>{}</td></tr> <tr> <td><span style='font-weight:bold'>End:</span></td><td>{}</td>  </tr>".format(len(self.dfs_02_01), self.dfs_concat_02_01['TIMESTAMP'].min(),self.dfs_concat_02_01['TIMESTAMP'].max())

                except:
                    print('erro')

    def _select_config(self, *args):
        with self.out_00:
            # self.dfs_01_01 = []
            # for i in self.select_meta.index:
            full_output_files = self.folder_path_ep.rglob('*{}*_full_output*.csv'.format(self.config_name[self.select_meta.index]))
            dfs_single_config = []
            for file in full_output_files:
                dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']}, usecols=self.ep_columns_filtered))

                # self.df_ep = pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']}, usecols=self.ep_columns_filtered)
            self.df_ep = pd.concat(dfs_single_config)

    def _button_plot(self, *args):
        with self.out_02:
            self.dfs_compare = pd.merge(left=self.dfs_concat_02_01, right=self.df_ep, how='outer', on='TIMESTAMP', suffixes=("_lf","_ep"))

            # self.date_range.options = self.dfs_compare['TIMESTAMP'].to_list()
            self.date_range.options = self.dfs_compare['TIMESTAMP'].dt.date.unique()
            self.hour_range.options = sorted(list(self.dfs_compare['TIMESTAMP'].dt.time.unique()))

            self.theta = np.linspace(0,360,36)
            theta1 = np.linspace(0,360,37)
            self.dfs_compare['wind_bin'] = pd.cut(x=self.dfs_compare['wind_dir'], bins=theta1)

            self.update_ep()

    def df_filter(self, *args):
        pass

    def update_ep(self, *args):
        self.fig_01.xaxis.axis_label = 'TIMESTAMP'
        self.source_ep.data = dict(x=self.dfs_compare['TIMESTAMP'],
                                   u=self.dfs_compare['u_unrot'], u_c=self.dfs_compare['u_rot'],
                                   v=self.dfs_compare['v_unrot'], v_c=self.dfs_compare['v_rot'],
                                   w=self.dfs_compare['w_unrot'], w_c=self.dfs_compare['w_rot'])
        push_notebook()

    def filter_date(self):
        with self.out_02:
            try:
                filter_date = self.dfs_compare[
                    (self.dfs_compare['TIMESTAMP'].dt.date>self.date_range.value[0]) &
                    (self.dfs_compare['TIMESTAMP'].dt.date<=self.date_range.value[1]) &
                    (self.dfs_compare['TIMESTAMP'].dt.time>self.hour_range.value[0]) &
                    (self.dfs_compare['TIMESTAMP'].dt.time<=self.hour_range.value[1])
                ]
            except:
                pass

        return filter_date

    def update_byDate(self, *args):
        with self.out_02:

        # self.filter_date()
            try:
                self.df_filter_date = self.filter_date()
                start_angle = np.arange(0,360,10)*np.pi/180
                end_angle = np.arange(10,370,10)*np.pi/180


                self.fig_03.title.text = 'Wind Direction from {} to {} ({} - {})'.format(self.date_range.value[0].strftime('%Y-%m-%d'), self.date_range.value[1].strftime('%Y-%m-%d'),self.hour_range.value[0].strftime('%H:%M'),self.hour_range.value[1].strftime('%H:%M'))
                self.source_ep2.data = dict(inner=[0 for i in range(36)],
                                            outer=self.df_filter_date.groupby(by='wind_bin').count()['wind_dir']/self.df_filter_date.groupby(by='wind_bin').count()['wind_dir'].max(),
                                            start=start_angle,
                                            end=end_angle)
                push_notebook()
            except:
                pass
