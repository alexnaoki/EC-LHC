import ipywidgets
import numpy as np
import pandas as pd
import pathlib
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, RangeTool, Select
from bokeh.layouts import gridplot, column, row


class view_files:
    def __init__(self):

        self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
 'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',
  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var']

        self.lf_columns_filtered = ['TIMESTAMP','Hs','u_star','Ts_stdev','Ux_stdev','Uy_stdev','Uz_stdev','Ux_Avg', 'Uy_Avg', 'Uz_Avg',
                                    'Ts_Avg', 'LE_wpl', 'Hc','H2O_mean', 'amb_tmpr_Avg', 'amb_press_mean', 'Tc_mean', 'rho_a_mean','CO2_sig_strgth_mean',
                                    'H2O_sig_strgth_mean','T_tmpr_rh_mean', 'e_tmpr_rh_mean', 'e_sat_tmpr_rh_mean', 'H2O_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                     'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
                                      'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)', 'precip_Tot', 'panel_tmpr_Avg',]
        self.TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

        output_notebook()

        self.tabs = ipywidgets.Tab([self.tab00(), self.tab01()])
        self.tabs.set_title(0, 'EP - Master Folder')
        self.tabs.set_title(1, 'LowFreq - Master Folder')

        self.column01 = Select(value='H', options=self.ep_columns_filtered)
        self.column02 = Select(value='Rn_Avg', options=self.lf_columns_filtered)

        # self.column01.on_change('value', self.ticker1_change)
        # self.column02.on_change('value', self.ticker2_change)


        self.source = ColumnDataSource(data=dict(date=[], c1=[], c2=[]))
        self.source_static = ColumnDataSource(data=dict(date=[], c1=[], c2=[]))

        # self.source.selected.on_change('indices', self._selection_change)


        tools = 'pan,wheel_zoom,xbox_select,reset'

        self.fig_corr = figure(plot_width=350, plot_height=350,tools='pan,wheel_zoom,box_select,reset')
        self.fig_corr.circle(x='c1', y='c2', source=self.source,selection_color="orange", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)

        self.ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
        self.ts1.line('date', 'c1', source=self.source_static)
        self.ts1.circle('date', 'c1', size=1, source=self.source, color=None, selection_color='orange')

        self.ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
        self.ts2.x_range = self.ts1.x_range
        self.ts2.line('date', 'c2', source=self.source_static)
        self.ts2.circle('date', 'c2', size=1, source=self.source, color=None, selection_color='orange')




        display(self.tabs)

        widgets = column(self.column01, self.column02)
        main_row = row(self.fig_corr, widgets)
        series = column(self.ts1, self.ts2)
        layout = column(main_row, series)

        show(layout, notebook_handle=True)

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
            try:
                # self.dropdown_yAxis_ep.options = self.ep_columns_filtered
                # self.dropdown_yAxis_ep.value = 'H'
                self.df_merged = pd.merge(self.df_ep,self.dfs_concat_02_01,how='outer', on='TIMESTAMP', suffixes=("_ep", "_lf"))
                # print(self.df_merged)
            except:
                print('erro')

    def ticker1_change(self, attrname, old, new):
        self.update()
        push_notebook()

    def ticker2_change(self, attrname, old, new):
        self.update()
        push_notebook()

    def update(self, selected=None):
        t1, t2 = self.column01.value, self.column02.value

        data = self.df_merge[[t1,t2]]
        self.source.data = data
        self.source_static.data = data
        push_notebook()

    def _selection_change(self, attrname, old, new):
        t1, t2 = self.column01.value, self.column02.value

        data = self.df_merged[[t1,t2]]

        selected = self.source.selected.indices
        if selected:
            data = data.iloc[selected, :]
        push_notebook()
