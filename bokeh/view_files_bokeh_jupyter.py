import ipywidgets
import numpy as np
import pandas as pd
import pathlib
from scipy.stats import linregress
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, RangeTool, Circle, Slope, Label, Legend, LegendItem, LinearColorMapper
from bokeh.layouts import gridplot, column, row
from bokeh.transform import transform




class view_files:
    def __init__(self):

        self.ep_columns_filtered = ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
 'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed', 'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',
  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var','H_strg','LE_strg']

        self.lf_columns_filtered = ['TIMESTAMP','Hs','u_star','Ts_stdev','Ux_stdev','Uy_stdev','Uz_stdev','Ux_Avg', 'Uy_Avg', 'Uz_Avg',
                                    'Ts_Avg', 'LE_wpl', 'Hc','H2O_mean', 'amb_tmpr_Avg', 'amb_press_mean', 'Tc_mean', 'rho_a_mean','CO2_sig_strgth_mean',
                                    'H2O_sig_strgth_mean','T_tmpr_rh_mean', 'e_tmpr_rh_mean', 'e_sat_tmpr_rh_mean', 'H2O_tmpr_rh_mean', 'RH_tmpr_rh_mean',
                                     'Rn_Avg', 'albedo_Avg', 'Rs_incoming_Avg', 'Rs_outgoing_Avg', 'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg',
                                      'Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)', 'precip_Tot', 'panel_tmpr_Avg']
        self.TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

        output_notebook()

        self.tabs = ipywidgets.Tab([self.tab00(), self.tab01(), self.tab02()])


        self.tabs.set_title(0, 'EP - Master Folder')
        self.tabs.set_title(1, 'LowFreq - Master Folder')
        self.tabs.set_title(2, 'Plot')

        self.source_ep = ColumnDataSource(data=dict(x=[], y=[], y2=[], date=[],time=[],et=[]))

        self.fig_01 = figure(title='EP', plot_height=250, plot_width=700, x_axis_type='datetime', tools=self.TOOLS)
        circle_ep = self.fig_01.circle(x='x', y='y', source=self.source_ep)


        self.fig_02 = figure(title='LF', plot_height=250, plot_width=700, x_axis_type='datetime', x_range=self.fig_01.x_range)
        circle_lf = self.fig_02.circle(x='x', y='y2', source=self.source_ep, color='red')

        self.fig_03 = figure(title='EP x LF',plot_height=500, plot_width=500)
        circle_teste = self.fig_03.circle(x='y2', y='y', source=self.source_ep, color='green', selection_color="green",selection_fill_alpha=0.3, selection_line_alpha=0.3,
                                          nonselection_fill_alpha=0.1,nonselection_fill_color="grey",nonselection_line_color="grey",nonselection_line_alpha=0.1)

        # self.fig_04 = figure(title='ET', plot_width=1200, plot_height=600)
        # colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
        # self.colorMapper = LinearColorMapper(palette=colors)
        # self.fig_04.rect(source=self.source_ep, x='date',y='time', fill_color=transform('et', self.colorMapper), line_color=None, width=1,height=1)
        # self.hm = self.fig_04.rect(source=self.source_ep, x='date',y='time', line_color=None, width=1,height=1)

        self.label = Label(x=1.1, y=18, text='teste', text_color='black')
        self.label2 = Label(x=1.1, y=10, text='teste2', text_color='black')
        self.label3 = Label(x=1.2, y=11, text='teste3', text_color='black')
        self.label4 = Label(x=1,y=11, text='teste4', text_color='black')
        # self.label5 = Label(x=1, y=11, text='teste5', text_color='black')
        self.fig_03.add_layout(self.label)
        self.fig_03.add_layout(self.label2)
        self.fig_03.add_layout(self.label3)
        self.fig_03.add_layout(self.label4)
        # self.fig_03.add_layout(self.label5)
        # self.label_teste = Label(x=0,y=0, text='fasdfasdfasdfasdfas', text_color='black')
        # self.fig_03.add_layout(self.label_teste)

        # self.source_ep.selected.on_change('indices', self.selection_change)
        # slope11_l = self.fig_03.line(color='orange', line_dash='dashed')
        slope_11 = Slope(gradient=1, y_intercept=0, line_color='orange', line_dash='dashed', line_width=3)
        self.fig_03.add_layout(slope_11)

        # self.slope_lin_label = self.fig_03.line(color='red', line_width=3)
        self.slope_linregress = Slope(gradient=1.3, y_intercept=0,line_color='red', line_width=3)
        self.fig_03.add_layout(self.slope_linregress)

        c = column([self.fig_01, self.fig_02])

        display(self.tabs)
        show(row(c, self.fig_03), notebook_handle=True)

    # def teste_apagar(self, attr, old,new):
    #     print(new)


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
            self.dropdown_yAxis_ep = ipywidgets.Dropdown(description='EP Y-Axis', options=self.ep_columns_filtered)

            self.dropdown_yAxis_lf = ipywidgets.Dropdown(description='LF Y-Axis', options=self.lf_columns_filtered)

            self.checkBox_EnergyBalance = ipywidgets.Checkbox(value=False, description='Energy Balance')


            # self.intSlider_flagFilter = ipywidgets.IntSlider(value=2, min=0, max=2, step=1, description='Flag Filter')
            self.selectionSlider_flagFilter = ipywidgets.SelectionSlider(options=[0,1,2,'All'], value='All', description='Flag Filter')


            self.checkBox_rainfallFilter = ipywidgets.Checkbox(value=False, description='Rainfall Filter')

            self.floatSlider_signalStrFilter = ipywidgets.FloatSlider(value=0, min=0, max=1, step=0.01, description='Signal Str Filter')

            self.selectionRangeSlider_date = ipywidgets.SelectionRangeSlider(options=[0,1], description='Date Range', layout=ipywidgets.Layout(width='500px'))

            self.selectionRangeSlider_hour = ipywidgets.SelectionRangeSlider(options=[0,1], description='Hour Range', layout=ipywidgets.Layout(width='500px'))


            self.button_plot = ipywidgets.Button(description='Plot')
            # self.button_plot.on_click(self.update_ep)
            self.button_plot.on_click(self._button_plot)

            controls_ep = [self.dropdown_yAxis_ep,
                           self.selectionSlider_flagFilter,
                           self.checkBox_rainfallFilter,
                           self.floatSlider_signalStrFilter,
                           self.checkBox_EnergyBalance,
                           self.selectionRangeSlider_date,
                           self.selectionRangeSlider_hour]
            for control in controls_ep:
                control.observe(self.update_ep, 'value')

            controls_lf = [self.dropdown_yAxis_lf]
            for control in controls_lf:
                # control.observe(self.update_lf, 'value')
                control.observe(self.update_ep, 'value')

            return ipywidgets.VBox([ipywidgets.HBox([self.dropdown_yAxis_ep, self.dropdown_yAxis_lf, self.checkBox_EnergyBalance]),
                                    ipywidgets.HBox([self.selectionSlider_flagFilter, self.checkBox_rainfallFilter, self.floatSlider_signalStrFilter]),
                                    self.selectionRangeSlider_date,
                                    self.selectionRangeSlider_hour,
                                    self.button_plot])


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
                dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']},keep_date_col=True, usecols=self.ep_columns_filtered))

                # self.df_ep = pd.read_csv(file, skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date', 'time']}, usecols=self.ep_columns_filtered)
            self.df_ep = pd.concat(dfs_single_config)
            # try:
            #     self.dropdown_yAxis_ep.options = self.ep_columns_filtered
            #     self.dropdown_yAxis_ep.value = 'H'
            # except:
            #     pass


    def filter_flag_ep(self):
        try:
            flag = self.dfs_compare[
                (self.dfs_compare['H2O_sig_strgth_mean'] >= self.floatSlider_signalStrFilter.value) &
                (self.dfs_compare['TIMESTAMP'].dt.date >= self.selectionRangeSlider_date.value[0]) &
                (self.dfs_compare['TIMESTAMP'].dt.date <= self.selectionRangeSlider_date.value[1]) &
                (self.dfs_compare['TIMESTAMP'].dt.time >= self.selectionRangeSlider_hour.value[0]) &
                (self.dfs_compare['TIMESTAMP'].dt.time <= self.selectionRangeSlider_hour.value[1])
            ]


        except:
            flag = self.dfs_compare[
                (self.dfs_compare['H2O_sig_strgth_mean'] >= self.floatSlider_signalStrFilter.value)
            ]

        if self.checkBox_rainfallFilter.value == True:
            flag = flag[flag['precip_Tot']==0]

        if self.checkBox_EnergyBalance.value == True:
            if self.selectionSlider_flagFilter.value in [0,1,2]:
                flag = flag[flag[['qc_H', 'qc_LE']].isin([self.selectionSlider_flagFilter.value]).sum(axis=1)==2]
            if self.selectionSlider_flagFilter.value == 'All':
                pass

        if self.checkBox_EnergyBalance.value == False:
            if self.selectionSlider_flagFilter.value in [0,1,2]:
                flag = flag[flag['qc_{}'.format(self.dropdown_yAxis_ep.value)]==self.selectionSlider_flagFilter.value]
                if self.selectionSlider_flagFilter.value == 'All':
                    pass



        return flag


    def _button_plot(self, *args):
        with self.out_02:
            self.dfs_compare = pd.merge(left=self.dfs_concat_02_01, right=self.df_ep, how='outer', on='TIMESTAMP', suffixes=("_lf","_ep"))

            self.selectionRangeSlider_date.options = self.dfs_compare['TIMESTAMP'].dt.date.unique()
            self.selectionRangeSlider_hour.options = sorted(list(self.dfs_compare['TIMESTAMP'].dt.time.unique()))
            # print(self.dfs_compare)
            # self.update_lf()
            # self.slope_linregress.gradient = 5
            self.update_ep()

    def update_ep(self, *args):
        self.df_filter_ep = self.filter_flag_ep()
        # self.source_ep.data = dict(x=self.df_filter_ep['TIMESTAMP'], y=self.df_filter_ep['{}'.format(self.dropdown_yAxis_ep.value)], y2=self.df_filter_ep['{}'.format(self.dropdown_yAxis_lf.value)])
        # self.fig_01.xaxis.axis_label = 'TIMESTAMP'
        # self.fig_01.yaxis.axis_label = '{}'.format(self.dropdown_yAxis_ep.value)


        if self.checkBox_EnergyBalance.value == True:
            self.source_ep.data = dict(x=self.df_filter_ep['TIMESTAMP'],
                                       y=self.df_filter_ep[['H', 'LE','H_strg','LE_strg']].sum(axis=1, min_count=1),
                                       y2=self.df_filter_ep['Rn_Avg']-self.df_filter_ep[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1))
            # self.hm.fill_color=transform('et', self.colorMapper)
            #self.df_filter_ep[['Rn_Avg', 'shf_Avg(1)']].sum(axis=1, min_count=1)
            #self.df_filter_ep[['H', 'LE']].sum(axis=1, min_count=1)
            self.fig_01.xaxis.axis_label = 'TIMESTAMP'
            self.fig_01.yaxis.axis_label = 'H + LE'

            self.fig_02.xaxis.axis_label = 'TIMESTAMP'
            self.fig_02.yaxis.axis_label = 'Rn - G'

            self.fig_03.yaxis.axis_label = 'H + LE'
            self.fig_03.xaxis.axis_label = 'Rn - G'

            # self.fig_04.x_range.factors = self.df_filter_ep['date'].unique()
            # self.fig_04.y_range.factors = self.df_filter_ep['time'].unique()
            # self.label.text = 'fasfdasfasfasfaf'

            self.df_corr = pd.DataFrame()
            self.df_corr['EP'] = self.df_filter_ep[['H','LE']].sum(axis=1, min_count=1)
            self.df_corr['EP'] = self.df_filter_ep[['H','LE','H_strg','LE_strg']].sum(axis=1, min_count=1)
            # self.df_corr['LF'] = self.df_filter_ep[['Rn_Avg', 'shf_Avg(2)']].sum(axis=1, min_count=1)
            self.df_corr['LF'] = self.df_filter_ep['Rn_Avg']-self.df_filter_ep[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1)
            self.label.text = 'Pearson: {:.4f}'.format(self.df_corr.corr(method='pearson')['LF'][0])
            self.df_corr.dropna(inplace=True)
            # linear_regression = linregress(x=self.df_corr['LF'], y=self.df_corr['EP'])
            linear_regression = linregress(x=self.df_corr['LF'], y=self.df_corr['EP'])

            x = np.array(self.df_corr['LF'].to_list())
            x1 = x[:, np.newaxis]
            fit_linear = np.linalg.lstsq(x1, self.df_corr['EP'], rcond=None)

            # pbias = 100*(self.df_corr['EP']-self.df_corr['LF']).sum()/self.df_corr['LF'].sum()

            # self.label5.text = 'PBIAS: {:.4f}'.format(pbias)
            self.slope_linregress.gradient = fit_linear[0][0]

            self.label2.text = 'Slope: {:.4f}'.format(fit_linear[0][0])
            # self.label2.text = 'R: {:.4f} '.format(linear_regression[2])
            self.label.x = np.nanmin(self.df_corr['LF'])
            self.label.y = np.nanmax(self.df_corr['EP'])
            self.label2.x = np.nanmin(self.df_corr['LF'])
            self.label2.y = np.nanmax(self.df_corr['EP']-0.1*np.nanmax(self.df_corr['EP']))
            self.label3.x = np.nanmin(self.df_corr['LF'])
            self.label3.y = np.nanmax(self.df_corr['EP']-0.2*np.nanmax(self.df_corr['EP']))
            self.label4.x = np.nanmin(self.df_corr['LF'])
            self.label4.y = np.nanmax(self.df_corr['EP']-0.3*np.nanmax(self.df_corr['EP']))
            # self.label5.x = np.nanmin(self.df_corr['LF'])
            # self.label5.y = np.nanmax(self.df_corr['EP']-0.4*np.nanmax(self.df_corr['EP']))

            # self.slope_linregress.gradient = linear_regression[0]
            # self.slope_linregress.y_intercept = linear_regression[1]
            self.label3.text = 'ET: {:.2f}'.format(self.df_filter_ep['ET'].sum()/2)
            self.label4.text = 'y = {:.4f}x + {:.4f}'.format(linear_regression[0], linear_regression[1])

            # self.slope_lin_label.legend_label = 'ok'
            # self.legend_fig03[0].label='ok'

        if self.checkBox_EnergyBalance.value == False:

            self.source_ep.data = dict(x=self.df_filter_ep['TIMESTAMP'], y=self.df_filter_ep['{}'.format(self.dropdown_yAxis_ep.value)], y2=self.df_filter_ep['{}'.format(self.dropdown_yAxis_lf.value)])
            self.fig_01.xaxis.axis_label = 'TIMESTAMP'
            self.fig_01.yaxis.axis_label = '{}'.format(self.dropdown_yAxis_ep.value)


        push_notebook()
