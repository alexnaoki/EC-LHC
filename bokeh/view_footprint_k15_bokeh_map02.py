from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import Slider, ColumnDataSource, Button, Tabs, Panel, DateSlider, Range1d, Div, TextInput, Select, Panel, DateRangeSlider,Legend, LegendItem,DatetimeTickFormatter,BasicTicker, LinearColorMapper,ColorBar
from bokeh.layouts import column, row, layout, gridplot
from bokeh.tile_providers import ESRI_IMAGERY, get_provider
from bokeh.transform import cumsum, transform
from bokeh.palettes import Spectral10

import sys, pathlib
sys.path.append(str(pathlib.Path(sys.path[0]).parent/'footprint'/'FFP_python_v1_4'))

from calc_footprint_FFP_adjusted01 import FFP
from calc_footprint_FFP_climatology_adjusted01 import FFP_climatology
import numpy as np
import pandas as pd
import datetime as dt
from shapely.geometry.polygon import Polygon
import rasterio
import rasterio.mask


class view_k15:
    def __init__(self):
        '''
        Visualizador do Footprint, tanto de um intervalo médio de análise como um range, utilizando o método por Kljun et al. (2015)
        "bokeh serve --show [Full Path deste arquivo .py]"
        '''
        # Colunas do FullOutput do EddyPro a serem utilizadas
        self.ep_columns_filtered = ['date','time','wind_dir', 'u_rot','L','v_var','u*','wind_speed']

        # Coordenadas x, y da torre IAB3 para web Marcator
        self.iab3_x_utm_webMarcator = -5328976.90
        self.iab3_y_utm_webMarcator = -2532052.38

        # Localização do arquivo tif retirado do mapbiomas no sistema de coordenadas Sirgas 2000 23S
        self.tif_file = r'C:\Users\User\git\EC-LHC\iab3_site\IAB1_SIRGAS_23S.tif'
        self.raster = rasterio.open(self.tif_file)

        # Coordenadas x,y da torre IAB3 para Sirgas 2000 23S
        self.iab3_x_utm_sirgas = 203917.07880027
        self.iab3_y_utm_sirgas = 7545463.6805863

        # Inicialização dos 3 tabs
        self.tabs = Tabs(tabs=[self.tab_01(), self.tab_02(), self.tab_03(), self.tab_04(), self.tab_05()])

        # Gerar servidor para rodar o programa
        curdoc().add_root(self.tabs)

    def tab_01(self):
        '''
        O tab_01 tem por objetivo inserir os dados do EddyPro
        '''
        # self.div_01 = Div(text=r'C:\Users\User\Mestrado\Dados_Processados\EddyPro_Fase01', width=500)
        self.div_01 = Div(text=r'G:\Meu Drive\USP-SHS\Resultados_processados\EddyPro_Fase010203', width=500)

        # Widgets e aplicação das funções no tab_01
        self.path_ep = TextInput(value='', title='EP Path:')
        self.path_ep.on_change('value', self._textInput)

        self.select_config = Select(title='Configs:', value=None, options=[])
        self.select_config.on_change('value', self._select_config)

        self.button_plot = Button(label='Plot')
        self.button_plot.on_click(self._button_plot)

        tab01 = Panel(child=column(self.div_01, self.path_ep, self.select_config, self.button_plot), title='EP Config')

        return tab01

    def tab_02(self):
        '''
        O tab_02 tem por objetivo a visualização do footprint por intervalo médio de análise
        '''
        # Range do x, y relativo ao web Mercator
        x_range = (-5332000, -5327000)
        y_range = (-2535000, -2530000)

        # Tile para display do mapa na figura
        tile_provider = get_provider(ESRI_IMAGERY)

        # Inicialização da Função de footprint de Kljun et al. (2015) por intervalo médio de análise
        self.k15_individual = FFP()

        self.div_02_01 = Div(text='Footprint por intervalo de 30 minutos através da metodologia Kljun et al. (2015)', width=500)

        # Widgets e aplicação da função
        self.datetime_slider = DateSlider(title='Datetime:',
                                          start=dt.datetime(2018,1,1,0,0),
                                          end=dt.datetime(2018,1,1,0,30),
                                          value=dt.datetime(2018,1,1,0,0),
                                          step=1000*60*30, format="%d/%m/%Y %H:%M")
        self.datetime_slider.on_change('value_throttled', lambda attr, old, new: self.update_ffp())

        self.div_02_02 = Div(text='Selecione os dados', width=500)

        # Figura 01 - Mapa e Footprint
        self.source_01 = ColumnDataSource(data=dict(xrs=[], yrs=[]))
        self.fig_01 = figure(title='Footprint K15', plot_height=400, plot_width=400, x_range=x_range, y_range=y_range)
        self.fig_01.add_tile(tile_provider)

        # Glyps da localização da torre IAB3 e contribuição de 90% do footprint
        iab03 = self.fig_01.circle([self.iab3_x_utm_webMarcator], [self.iab3_y_utm_webMarcator], color='red')
        mlines = self.fig_01.multi_line(xs='xrs', ys='yrs', source=self.source_01, color='red', line_width=1)

        # Figura 02 - Categoria do tipo de vegetação pelo footprint
        self.source_01_02 = ColumnDataSource(data=dict(angle=[], color=[], significado=[]))
        self.fig_01_02 = figure(plot_height=300, plot_width=500, x_range=(-1.6,1.4), toolbar_location=None)
        wedge_significado = self.fig_01_02.annular_wedge(x=-1, y=0, inner_radius=0.3, outer_radius=0.45,
                                                         start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                                                         line_color='white', fill_color='color', legend_field='significado', source=self.source_01_02)
        self.fig_01_02.axis.axis_label=None
        self.fig_01_02.axis.visible=False
        self.fig_01_02.grid.grid_line_color = None
        self.fig_01_02.outline_line_color = None

        # Tabela para basic stats
        self.div_02_03 = Div(text='''
                          <div class="header">
                          <h1>Basic Stats</h1>
                          <table border="1"><tbody><tr>
                          <td><b>Floresta (#)</b>&nbsp;</td>
                          <td><b>Outros (#)</b></td>
                          <td><b>Aceitação (%)</b></td>
                          </tr><tr>
                          <td>&nbsp;0</td>
                          <td>0</td>
                          <td>0</td>
                          </tr></tbody></table>
                          </div>''', width=500)

        # Criação do tab02
        tab02 = Panel(child=column(self.div_02_01,
                                   self.datetime_slider,
                                   self.div_02_02,
                                   row(self.fig_01, column(self.div_02_03,
                                                           self.fig_01_02))), title='Footprint per time')

        return tab02

    def tab_03(self):
        '''
        O tab_03 tem por objetivo a visualização do footprint por range de intervalo médio de análise e verificação da direção do vento
        '''
        # Range do x, y relativo ao web Mercator
        x_range = (-5332000, -5327000)
        y_range = (-2535000, -2530000)

        # Tile para display do mapa na figura
        tile_provider = get_provider(ESRI_IMAGERY)

        # Inicialização da Função de footprint de Kljun et al. (2015) por range de intervalo médio de análise
        self.k15_climatology = FFP_climatology()

        self.div_03 = Div(text='Footprint acumulado através da metodologia Kljun et al. (2015) e direção do vento', width=500)

        # Widgets e aplicação da função
        self.date_range = DateRangeSlider(title='Date', start=dt.datetime(2018,1,1),
                                          end=dt.datetime(2019,1,1),
                                          value=(dt.datetime(2018,1,1), dt.datetime(2019,1,1)),
                                          step=24*60*60*1000, format="%d/%m/%Y")

        self.time_range = DateRangeSlider(title='Time', start=dt.datetime(2012,1,1,0,0),
                                          end=dt.datetime(2012,1,1,23,30),
                                          value=(dt.datetime(2012,1,1,0,0), dt.datetime(2012,1,1,0,30)),
                                          step=30*60*1000, format='%H:%M')

        self.date_range.on_change('value', lambda attr,old,new:self.update_windDir())
        self.time_range.on_change('value', lambda attr,old,new:self.update_windDir())

        # A figura responsável pelo footprint é apenas atualizada após o click deste botão
        self.button_update_ffp = Button(label='Update Plot', width=500)
        self.button_update_ffp.on_click(self._button_update_ffp)

        # Figura 03 - Mapa e Footprint acumulado
        self.source_02 = ColumnDataSource(data=dict(xrs=[], yrs=[]))
        self.fig_02 = figure(title='Footprint K15 acumulado', plot_height=400, plot_width=400, x_range=x_range, y_range=y_range)
        self.fig_02.add_tile(tile_provider)
        self.fig_02.title.align = 'center'
        self.fig_02.title.text_font_size = '20px'

        # Glyphs da localização da torre IAB3 e contribuição de 90% do footprint e as legendas
        iab03 = self.fig_02.circle([self.iab3_x_utm_webMarcator], [self.iab3_y_utm_webMarcator], color='red')
        mlines = self.fig_02.multi_line(xs='xrs', ys='yrs', source=self.source_02, color='red', line_width=1)
        legend = Legend(items=[
            LegendItem(label='Torre IAB3', renderers=[iab03], index=0),
            LegendItem(label='Footprint Kljun et al. (90%)', renderers=[mlines], index=1)
        ])
        self.fig_02.add_layout(legend)

        # Figura 04 - Rosa dos ventos, com a direção dos ventos corrigida
        self.source_02_02 = ColumnDataSource(data=dict(inner=[0], outer=[1], start=[0],end=[2]))
        self.fig_02_02 = figure(title='Direção do vento', plot_height=400, plot_width=400, toolbar_location=None, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2))
        self.fig_02_02.axis.visible = False
        self.fig_02_02.axis.axis_label = None
        self.fig_02_02.grid.grid_line_color = None
        self.fig_02_02.outline_line_color = None
        self.fig_02_02.title.align = 'center'
        self.fig_02_02.title.text_font_size = '20px'

        # Glyphs da direção do vento e glyphs auxiliares para o grid dessa figura
        wedge = self.fig_02_02.annular_wedge(x=0, y=0, inner_radius='inner', outer_radius='outer', start_angle='start', end_angle='end', color='#FF00FF', source=self.source_02_02)
        circle = self.fig_02_02.circle(x=0, y=0, radius=[0.25,0.5,0.75], fill_color=None,line_color='white')
        circle2 = self.fig_02_02.circle(x=0, y=0, radius=[1], fill_color=None, line_color='grey')
        self.fig_02_02.annular_wedge(x=0, y=0, inner_radius='inner', outer_radius='outer', start_angle='start', end_angle='end', line_color='white',fill_color=None, line_width=1,source=self.source_02_02)

        self.date_range.on_change('value', lambda attr,old,new:self.update_windDir())
        self.time_range.on_change('value', lambda attr,old,new:self.update_windDir())

        # Figura 05 - Categoria do tipo de vegetação pelo footprint
        self.source_02_03 = ColumnDataSource(data=dict(angle=[], color=[], significado=[]))
        self.fig_02_03 = figure(plot_height=300, plot_width=500, x_range=(-1.6,1.4), toolbar_location=None)
        self.fig_02_03.axis.axis_label=None
        self.fig_02_03.axis.visible=False
        self.fig_02_03.grid.grid_line_color = None
        self.fig_02_03.outline_line_color = None

        # Glyph das categorias de vegetação
        wedge_significado = self.fig_02_03.annular_wedge(x=-1, y=0, inner_radius=0.3, outer_radius=0.45,
                                                         start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                                                         line_color='white', fill_color='color', legend_field='significado', source=self.source_02_03)

        # Tabela para basic stats
        self.div_03_02 = Div(text='''
                          <div class="header">
                          <h2>Basic Stats</h2>
                          <table border="1"><tbody><tr>
                          <td><b>Floresta (#)</b>&nbsp;</td>
                          <td><b>Outros (#)</b></td>
                          <td><b>Aceitação (%)</b></td>
                          </tr><tr>
                          <td>&nbsp;0</td>
                          <td>0</td>
                          <td>0</td>
                          </tr></tbody></table>
                          </div>''', width=400,sizing_mode="stretch_width")


        # Organização do tab
        widgets = column(self.date_range,self.time_range,self.button_update_ffp)

        tab03 = Panel(child=column(self.div_03,
                                   self.date_range,
                                   self.time_range,
                                   self.button_update_ffp,
                                   row(self.fig_02, self.fig_02_02, column(self.div_03_02,
                                                                           self.fig_02_03))), title='Footprint per range')
        # layout03 = layout([[self.div_03],
        #                    [widgets],
        #                    [self.fig_02, self.fig_02_02, column(self.div_03_02)]]s)
        # tab03 = Panel(child=layout03, title='Footprint per range')
        return tab03

    def tab_04(self):
        self.div_04_01 = Div(text='Para o funcionamento dessa aba é necessário o processamento dos dados e criação de uma arquivo CSV (esquerda) ou indicação de um arquivo já processado (direita). Para isso, utiliza-se a metodologia por Kljun et al. (2015) com uma área de contribuição de até 90%. (Recomenda-se entre 80% ~ 90%)', width=1200)

        self.datetime_range = DateRangeSlider(title='Date', start=dt.datetime(2018,1,1),
                                              end=dt.datetime(2019,1,1),
                                              value=(dt.datetime(2018,1,1), dt.datetime(2019,1,1)),
                                              step=30*60*1000, format="%d/%m/%Y %H:%M")
        # self.datetime_range.on_change('value', lambda attr, old,new:self._teste1())
        self.path_download = TextInput(value='')
        self.button_download = Button(label='Download', width=150, button_type='danger')
        self.button_download.on_click(self._button_download)

        self.div_04_02 = Div(text=r'''C:\Users\User\Mestrado\Testes\classification_pixel_2018-10-05-00-30to2020-07-15-00-00_pf_90.csv C:\Users\User\Mestrado\Testes\classification_pixel_2018-10-05-00-30to2020-07-15-00-00_dr_90.csv C:\Users\User\Mestrado\Testes\classification_pixel_2018-10-05-00-30to2020-07-15-00-00_pf_80.csv''')

        self.path_footprintStats_k15 = TextInput(value='')
        self.button_update_footprintstats = Button(label='Update', button_type='success')
        self.button_update_footprintstats.on_click(self._button_update_heatmap)


        self.source_04 = ColumnDataSource(data=dict(date=[],time=[],classification_pixel=[], wind_dir=[], florest_s_percentage=[],pasto_percentage=[],code03=[],code04=[],code15=[],resto_code=[]))

        self.fig_04 = figure(title='Aceitação -> # Floresta Natural / # Total', plot_height=350, plot_width=1200,x_axis_type='datetime', y_axis_type='datetime', tools="hover,pan,wheel_zoom,box_zoom,reset,box_select,tap")
        self.fig_04.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_04.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])
        self.fig_04.axis.axis_line_color = None
        self.fig_04.axis.major_tick_line_color = None

        self.color_mapper_pixels = LinearColorMapper(palette="Cividis256")
        self.fig_04.rect(x='date',
                         y='time',
                         fill_color=transform('classification_pixel', self.color_mapper_pixels),
                         source=self.source_04,
                         width=1000*60*60*24, height=1000*60*30, line_color=None)
        color_bar = ColorBar(color_mapper=self.color_mapper_pixels, ticker=BasicTicker(desired_num_ticks=len(Spectral10)),label_standoff=6, border_line_color=None, location=(0,0))
        self.fig_04.add_layout(color_bar, 'right')

        self.fig_04.hover.tooltips = [
                                      ("Acception", "@classification_pixel"),
                                      ("Wind Direction", "@wind_dir")
        ]
        self.fig_04.xaxis.major_label_orientation = 1


        self.fig_05 = figure(title='Subset -> Proporção Formação Savânica x Formação Florestal', plot_height=225, plot_width=600, x_axis_type='datetime', y_axis_type='datetime', tools="hover,pan,wheel_zoom,box_zoom,reset,box_select,tap",
                             x_range=self.fig_04.x_range, y_range=self.fig_04.y_range)
        self.fig_05.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_05.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])
        self.fig_05.axis.axis_line_color = None
        self.fig_05.axis.major_tick_line_color = None

        self.color_mapper_florest = LinearColorMapper(palette='Viridis256')
        self.fig_05.rect(x='date',
                         y='time',
                         fill_color=transform('florest_s_percentage', self.color_mapper_florest),
                         source=self.source_04,
                         width=1000*60*60*24, height=1000*60*30,  line_color=None)
        color_bar02 = ColorBar(color_mapper=self.color_mapper_florest, ticker=BasicTicker(desired_num_ticks=len(Spectral10)), label_standoff=6, border_line_color=None, location=(0,0))
        self.fig_05.add_layout(color_bar02, 'right')

        self.fig_05.hover.tooltips = [
                                      ("Florest S Percentage", "@florest_s_percentage"),
                                      ("Wind Direction", "@wind_dir"),
                                      ("Code03","@code03"),
                                      ("Code04","@code04"),
                                      ("Resto code","@resto_code")
        ]

        self.fig_06 = figure(title='Subset -> Proporção Pasto x Resto', plot_height=225, plot_width=600, x_axis_type='datetime', y_axis_type='datetime', tools="hover,pan,wheel_zoom,box_zoom,reset,box_select,tap",
                             x_range=self.fig_04.x_range, y_range=self.fig_04.y_range)
        self.fig_06.xaxis[0].formatter = DatetimeTickFormatter(days=["%d/%m/%Y"])
        self.fig_06.yaxis[0].formatter = DatetimeTickFormatter(days=["%H:%M"], hours=["%H:%M"])
        self.fig_06.axis.axis_line_color = None
        self.fig_06.axis.major_tick_line_color = None

        self.color_mapper_pasto = LinearColorMapper(palette="Viridis256")
        self.fig_06.rect(x='date',
                         y='time',
                         fill_color=transform('pasto_percentage', self.color_mapper_pasto),
                         source=self.source_04,
                         width=1000*60*60*24, height=1000*60*30,  line_color=None)
        color_bar03 = ColorBar(color_mapper=self.color_mapper_pasto, ticker=BasicTicker(desired_num_ticks=len(Spectral10)), label_standoff=6, border_line_color=None, location=(0,0))
        self.fig_06.add_layout(color_bar02, 'right')

        self.fig_06.hover.tooltips = [
                                      ("Pasto Percentage", "@pasto_percentage"),
                                      ("Wind Direction", "@wind_dir"),
                                      ("Code15", "@code15"),
                                      ("Resto code","@resto_code")
        ]


        self.fig_04.toolbar.autohide = True
        self.fig_05.toolbar.autohide = True
        self.fig_06.toolbar.autohide = True

        tab04 = Panel(child=column(self.div_04_01,
                                   self.datetime_range,
                                   row(column(Div(text='Insert Folder to Download <b>(if not found)</b>:'),row(self.path_download, self.button_download)),column(Div(text='File path FootprintStats K15:'),row(self.path_footprintStats_k15, self.button_update_footprintstats)),),
                                   self.div_04_02,
                                   layout([[self.fig_04],[self.fig_05, self.fig_06]])), title='Heatmap')

        return tab04

    def tab_05(self):
        self.div_05 = Div(text='Footprint per acceptance', width=500)

        self.slider_footprint_acceptance = Slider(start=0, end=1, value=0, step=0.01, title='Footprint Acceptance')

        self.button_update_ffp_acceptance = Button(label='Update Plot', width=500)
        self.button_update_ffp_acceptance.on_click(self._button_update_ffp)

        # Range do x, y relativo ao web Mercator
        x_range = (-5332000, -5327000)
        y_range = (-2535000, -2530000)

        # Tile para display do mapa na figura
        tile_provider = get_provider(ESRI_IMAGERY)

        self.source_05 = ColumnDataSource(data=dict(xrs=[], yrs=[]))
        self.fig_07 = figure(title='Footprint K15 acceptance', plot_height=400, plot_width=400, x_range=x_range, y_range=y_range)
        self.fig_07.add_tile(tile_provider)

        iab03 = self.fig_07.circle([self.iab3_x_utm_webMarcator], [self.iab3_y_utm_webMarcator], color='red')
        mlines = self.fig_07.multi_line(xs='xrs', ys='yrs', source=self.source_05, color='red', line_width=1)

        legend = Legend(items=[
            LegendItem(label='Torre IAB3', renderers=[iab03], index=0),
            LegendItem(label='Footprint Kljun et al. (10% ~ 90%)', renderers=[mlines], index=1)
        ])
        self.fig_07.add_layout(legend)

        tab05 = Panel(child=column(self.div_05,
                                   self.slider_footprint_acceptance,
                                   self.button_update_ffp_acceptance,
                                   self.fig_07), title='Footprint per acceptance')
        return tab05

    def _textInput(self, attr, old, new):
        '''
        Função para ler o arquivo Readme.txt (Metafile) do FullOutput do EddyPro
        '''
        if self.tabs.active == 0:
            try:
                self.folder_path_ep = pathlib.Path(new)
                readme = self.folder_path_ep.rglob('Readme.txt')
                readme_df = pd.read_csv(list(readme)[0], delimiter=',')
                temp_list = [row.to_list() for i,row in readme_df[['rotation', 'lowfrequency', 'highfrequency','wpl','flagging','name']].iterrows()]
                a = []
                for i in temp_list:
                    a.append('Rotation:{} |LF:{} |HF:{} |WPL:{} |Flag:{}| Name:{}'.format(i[0],i[1],i[2],i[3],i[4],i[5]))
                self.select_config.options = a
            except:
                print('erro text input readme')

    def _select_config(self, attr, old, new):
        '''
        Função para ler um arquivo específico do FullOutput do EddyPro
        '''
        print(new)
        full_output_files = self.folder_path_ep.rglob('*{}*_full_output*.csv'.format(new[-3:]))
        dfs_single_config = []
        for file in full_output_files:
            dfs_single_config.append(pd.read_csv(file, skiprows=[0,2], na_values=-9999,
                                                 parse_dates={'TIMESTAMP':['date', 'time']},
                                                 keep_date_col=True,
                                                 usecols=self.ep_columns_filtered))
        self.df_ep = pd.concat(dfs_single_config)
        self.df_ep.dropna(inplace=True)
        print('ok')

    def update_ffp(self):
        '''
        Função para aplicação do footprint, a depender do tab selecionado
        '''
        if self.tabs.active == 1:
            # Transformação de float do widget para datetime
            datetime = dt.datetime.utcfromtimestamp(self.datetime_slider.value/1000)

            # Filtração por datetime do widget
            inputs_to_k15 = self.df_ep.loc[self.df_ep['TIMESTAMP']==datetime, ['u_rot', 'L', 'u*','v_var','wind_dir_compass']]
            print(inputs_to_k15)

            # Atualização da tabela com os inputs para a função do método de footprint por Kljun et al. (2015)
            self.div_02_02.text = '''
            <table border="2"><tbody><tr><td>&nbsp;zm</td><td>umean</td><td>h</td><td>ol</td><td>sigmav</td><td>ustar</td><td>wind_dir_compass</td>
    		</tr><tr>
    			<td>&nbsp;{:.3f}</td><td>{:.3f}&nbsp;</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>&nbsp;{:.3f}</td>
    		</tr></tbody></table>'''.format(9,
                                      inputs_to_k15['u_rot'].values[0],
                                      1000,
                                      inputs_to_k15['L'].values[0],
                                      inputs_to_k15['v_var'].values[0],
                                      inputs_to_k15['u*'].values[0],
                                      inputs_to_k15['wind_dir_compass'].values[0])

            # Output para o footprint de Kljun et a. (2015)
            out = self.k15_individual.output(zm=9,
                                             umean=inputs_to_k15['u_rot'].values[0],
                                             h=1000,
                                             ol=inputs_to_k15['L'].values[0],
                                             sigmav=inputs_to_k15['v_var'].values[0],
                                             ustar=inputs_to_k15['u*'].values[0],
                                             wind_dir=inputs_to_k15['wind_dir_compass'].values[0],
                                             rs=[0.9], crop=False, fig=False)

            # Criação do polígono do footprint 90% para Sirgas 2000 utm 23S com o ponto de referência a torre IAB3
            poly = [(i+self.iab3_x_utm_sirgas, j+self.iab3_y_utm_sirgas) for i, j in zip(out[8][-1], out[9][-1])]
            poly_shp = Polygon(poly)

            # Mask utilzindo o arquivo tif e o polígono (ambos no mesmo sistema de coordenadas)
            mask_teste = rasterio.mask.mask(self.raster, [poly_shp], crop=True, invert=False)

            # Contabilização e contagem do Mask
            unique, counts = np.unique(mask_teste[0], return_counts=True)

            # Função para calcular estatísticas básicas do Mask
            simplified_stats = self.stats_pixel(unique, counts)

            # Atualização da tabela com os dados do basic stats
            self.div_02_03.text = '''
            <div class="header">
            <h1>Basic Stats</h1><hr><table border="1"><tbody><tr>
            <td>Floresta (#)&nbsp;</td>
            <td>Outros (#)</td>
            <td>Aceitação (%)</td>
    		</tr><tr>
            <td>&nbsp;{}</td>
            <td>{}</td>
            <td>{:.2f}</td>
    		</tr></tbody></table>
            </div>'''.format(simplified_stats[0],
                             simplified_stats[1],
                             100*simplified_stats[0]/(simplified_stats[0]+simplified_stats[1]))


            # Transformação das coordenadas de footprint de Kljun et al. (2015) [0,0] para Web Marcator
            x_webMarcator = []
            y_webMarcator = []
            for i,j in zip(out[8], out[9]):
                x_webMarcator.append(list(np.array(i)+self.iab3_x_utm_webMarcator))
                y_webMarcator.append(list(np.array(j)+self.iab3_y_utm_webMarcator))
            # Atualização do source
            self.source_01.data = dict(xrs=x_webMarcator, yrs=y_webMarcator)

        if self.tabs.active == 2:
            # Transformação de float do widget para datetime
            start_date = dt.datetime.utcfromtimestamp(self.date_range.value[0]/1000).date()
            end_date = dt.datetime.utcfromtimestamp(self.date_range.value[1]/1000).date()

            start_time = dt.datetime.utcfromtimestamp(self.time_range.value[0]/1000).time()
            end_time = dt.datetime.utcfromtimestamp(self.time_range.value[1]/1000).time()

            # Filtração por datetime dos widgets
            inputs_to_k15 = self.df_ep.loc[
                (self.df_ep['TIMESTAMP'].dt.date >= start_date) &
                (self.df_ep['TIMESTAMP'].dt.date <= end_date) &
                (self.df_ep['TIMESTAMP'].dt.time >= start_time) &
                (self.df_ep['TIMESTAMP'].dt.time <= end_time),
                ['u_rot','L','u*', 'v_var','wind_dir_compass']
            ]
            print(inputs_to_k15)

            # Output para o footprint de Kljun et al. (2015)
            # Para mudar a contribuição é necessário alterar o rs
            out = self.k15_climatology.output(zm=9,
                                              umean=inputs_to_k15['u_rot'].to_list(),
                                              h=[1000 for i in range(len(inputs_to_k15['u_rot'].to_list()))],
                                              ol=inputs_to_k15['L'].to_list(),
                                              sigmav=inputs_to_k15['v_var'].to_list(),
                                              ustar=inputs_to_k15['u*'].to_list(),
                                              wind_dir=inputs_to_k15['wind_dir_compass'].to_list(),
                                              rs=[0.9], crop=False, fig=False)

            # Criação do polígono do footprint 90% para Sirgas 2000 utm 23S com o ponto de referência a torre IAB3
            print(np.shape(out['xr']), np.shape(out['yr']))
            print('XR', out['xr'])
            print('YR', out['yr'])
            poly = [(i+self.iab3_x_utm_sirgas, j+self.iab3_y_utm_sirgas) for i, j in zip(out['xr'][-1], out['yr'][-1])]
            poly_shp = Polygon(poly)

            # mask utilizando o arquivo tif e o polígono (ambos no mesmo sistema de coordenadas)
            mask_teste = rasterio.mask.mask(self.raster, [poly_shp], crop=True, invert=False)

            # Contabilização e contagem do Mask
            unique, counts = np.unique(mask_teste[0], return_counts=True)

            # Função para calcular estatísticas básicas do Mask
            simplified_stats = self.stats_pixel(unique, counts)

            # Atualização da tabela com os dados do basic stats
            self.div_03_02.text = '''
            <div class="header">
            <h1>Basic Stats</h1><hr><table border="1"><tbody><tr>
            <td>Floresta (#)&nbsp;</td>
            <td>Outros (#)</td>
            <td>Aceitação (%)</td>
    		</tr><tr>
            <td>&nbsp;{}</td>
            <td>{}</td>
            <td>{:.2f}</td>
    		</tr></tbody></table>
            </div>'''.format(simplified_stats[0],
                             simplified_stats[1],
                             100*simplified_stats[0]/(simplified_stats[0]+simplified_stats[1]))

            # Transformação das coordenadas de footprint de Kljun et al. (2015) [0,0] para Web Marcator
            # x_webMarcator = list(np.array(out['xr'][-1]) + self.iab3_x_utm_webMarcator)
            # y_webMarcator = list(np.array(out['yr'][-1]) + self.iab3_y_utm_webMarcator)
            x_webMarcator = []
            y_webMarcator = []
            for i,j in zip(out['xr'], out['yr']):
                x_webMarcator.append(list(np.array(i)+self.iab3_x_utm_webMarcator))
                y_webMarcator.append(list(np.array(j)+self.iab3_y_utm_webMarcator))

            # Atualização do source
            self.source_02.data = dict(xrs=x_webMarcator, yrs=y_webMarcator)

        if self.tabs.active == 4:
            print(self.slider_footprint_acceptance.value)
            df_footprint_acceptance = self.df_footprintstats.copy()
            df_footprint_acceptance_filter = df_footprint_acceptance.loc[df_footprint_acceptance['classification_percentage']>=self.slider_footprint_acceptance.value]

            # df_ep_footprint_acceptance = pd.merge(left=self.df_ep, right=df_footprint_acceptance_filter, on='TIMESTAMP', how='inner')

            # print(df_ep_footprint_acceptance.columns)

            out = self.k15_climatology.output(zm=9,
                                              umean=df_footprint_acceptance_filter['u_rot'].to_list(),
                                              h=[1000 for i in range(len(df_footprint_acceptance_filter['u_rot'].to_list()))],
                                              ol=df_footprint_acceptance_filter['L'].to_list(),
                                              sigmav=df_footprint_acceptance_filter['v_var'].to_list(),
                                              ustar=df_footprint_acceptance_filter['u*'].to_list(),
                                              wind_dir=df_footprint_acceptance_filter['wind_dir_compass'].to_list(),
                                              rs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], crop=False,fig=False)
            print(np.shape(out['xr']), np.shape(out['yr']))
            print('XR', out['xr'])
            print('YR', out['yr'])

            # Transformação das coordenadas de footprint de Kljun et al. (2015) [0,0] para Web Marcator
            # x_webMarcator = list(np.array(out['xr'][-1]) + self.iab3_x_utm_webMarcator)
            # y_webMarcator = list(np.array(out['yr'][-1]) + self.iab3_y_utm_webMarcator)
            x_webMarcator = []
            y_webMarcator = []
            for i,j in zip(out['xr'], out['yr']):
                x_webMarcator.append(list(np.array(i)+self.iab3_x_utm_webMarcator))
                y_webMarcator.append(list(np.array(j)+self.iab3_y_utm_webMarcator))

            self.source_05.data = dict(xrs=x_webMarcator, yrs=y_webMarcator)

    def update_windDir(self):
        # Transformação do float para datetime
        start_date = dt.datetime.utcfromtimestamp(self.date_range.value[0]/1000).date()
        end_date = dt.datetime.utcfromtimestamp(self.date_range.value[1]/1000).date()

        start_time = dt.datetime.utcfromtimestamp(self.time_range.value[0]/1000).time()
        end_time = dt.datetime.utcfromtimestamp(self.time_range.value[1]/1000).time()

        # Criação do vetor de intervalos de ângulos e iniciando como 0° no Norte (posição vertical)
        start_angle = np.arange(0,360,10)*np.pi/180 + 90*np.pi/180
        end_angle = np.arange(10,370,10)*np.pi/180 + 90*np.pi/180

        # Filtração por datetime
        filter = self.df_ep[(self.df_ep['TIMESTAMP'].dt.date >= start_date) &
                            (self.df_ep['TIMESTAMP'].dt.date <= end_date) &
                            (self.df_ep['TIMESTAMP'].dt.time >= start_time) &
                            (self.df_ep['TIMESTAMP'].dt.time <= end_time)]

        # Atualização do source, sendo que é contado o número por bins
        self.source_02_02.data = dict(inner=[0 for i in range(36)],
                                   outer=filter.groupby(by='wind_bin').count()['wind_dir_compass'][::-1]/filter.groupby(by='wind_bin').count()['wind_dir_compass'].max(),
                                   start=start_angle,
                                   end=end_angle)

    def _button_plot(self):
        # Atualização dos ranges do widget
        self.datetime_slider.start = self.df_ep['TIMESTAMP'].min()
        self.datetime_slider.end = self.df_ep['TIMESTAMP'].max()
        self.datetime_slider.value = self.df_ep['TIMESTAMP'].min()

        self.datetime_range.start = self.df_ep['TIMESTAMP'].min()
        self.datetime_range.end = self.df_ep['TIMESTAMP'].max()
        self.datetime_range.value = (self.df_ep['TIMESTAMP'].min(), self.df_ep['TIMESTAMP'].max())

        # Função para corrigir direção do vento
        self._adjust_wind_direction()

        # Criação de bins para discretização da direção do vento
        self.theta = np.linspace(0, 360, 36)
        theta01 = np.linspace(0, 360, 37)
        # print(theta01)
        self.df_ep['wind_bin'] = pd.cut(x=self.df_ep['wind_dir_compass'], bins=theta01)
        # print(self.df_ep['wind_bin'])

        self.date_range.start = self.df_ep['date'].min()
        self.date_range.end = self.df_ep['date'].max()
        self.date_range.value = (self.df_ep['date'].min(), self.df_ep['date'].max())

    def _button_update_ffp(self):
        self.update_ffp()


    def _adjust_wind_direction(self):
        '''
        Informação sobre essa transformação se encontra na ../info/wind_direction.md
        '''
        self.df_ep['wind_dir_sonic'] = 360 - self.df_ep['wind_dir']
        azimute = 135.1
        self.df_ep['wind_dir_compass'] = (360 + azimute - self.df_ep['wind_dir_sonic']).apply(lambda x: x-360 if x>=360 else x)


    def stats_pixel(self, unique, counts):
        '''
        Output dos stats_pixel varia de acordo com o tab selecionado
        '''
        if self.tabs.active == 1:
            # Dicionário e lista do significado do pixel do tif. Não foi inserido 255.
            significado_pixel = {3: 'Floresta Natural => Formação Florestal',
                                 4: 'Floesta Natural => Formação Savânica',
                                 9: 'Floresta Plantada',
                                 12: 'Formação Campestre/Outra formação não Florestal',
                                 15: 'Pastagem',
                                 19: 'Agricultura => Cultivo Anual e Perene',
                                 20: 'Agricultura => Cultivo Semi-Perene',
                                 24: 'Infraestrutura Urbana',
                                 25: 'Outra área não Vegetada',
                                 33: "Corpo d'água"}
            significado_pixel_lista = ['Floresta Natural (Formação Florestal)', 'Floesta Natural (Formação Savânica)',
                                       'Floresta Plantada', 'Formação Campestre', 'Pastagem', 'Agricultura (Cultivo Anual e Perene)',
                                       'Agricultura (Cultivo Semi-Perene)', 'Infraestrutura Urbana', 'Outra área não Vegetada', "Corpo d'água"]

            pixel_dict = dict(zip(unique, counts))

            # A partir do dicionário contiver counts será inserido na lista, caso não contiver 0 será aplicado.
            pixel_simplified = []
            for i in significado_pixel:
                try:
                    pixel_simplified.append(pixel_dict[i])
                except:
                    pixel_simplified.append(0)

            # Pixels de Floresta Natural serão somados, enquanto e o resto também
            pixel_floresta = pixel_simplified[0] + pixel_simplified[1]
            pixel_resto = sum(pixel_simplified[2:])

            # Criação do DataFrame e calculo do ângulo para o wedge da figura 02
            df = pd.DataFrame({'significado': significado_pixel_lista, 'value':pixel_simplified})
            df['angle'] = df['value']/df['value'].sum() * 2*np.pi

            self.source_01_02.data = dict(angle=df['angle'],
                                       color=Spectral10,
                                       significado=df['significado'])
            return pixel_floresta, pixel_resto

        if self.tabs.active == 2:
            # Dicionário e lista do significado do pixel do tif. Não foi inserido 255.
            significado_pixel = {3: 'Floresta Natural => Formação Florestal',
                                 4: 'Floesta Natural => Formação Savânica',
                                 9: 'Floresta Plantada',
                                 12: 'Formação Campestre/Outra formação não Florestal',
                                 15: 'Pastagem',
                                 19: 'Agricultura => Cultivo Anual e Perene',
                                 20: 'Agricultura => Cultivo Semi-Perene',
                                 24: 'Infraestrutura Urbana',
                                 25: 'Outra área não Vegetada',
                                 33: "Corpo d'água"}
            significado_pixel_lista = ['Floresta Natural (Formação Florestal)', 'Floesta Natural (Formação Savânica)',
                                       'Floresta Plantada', 'Formação Campestre', 'Pastagem', 'Agricultura (Cultivo Anual e Perene)',
                                       'Agricultura (Cultivo Semi-Perene)', 'Infraestrutura Urbana', 'Outra área não Vegetada', "Corpo d'água"]

            pixel_dict = dict(zip(unique, counts))

            # A partir do dicionário contiver counts será inserido na lista, caso não contiver 0 será aplicado.
            pixel_simplified = []
            for i in significado_pixel:
                try:
                    pixel_simplified.append(pixel_dict[i])
                except:
                    pixel_simplified.append(0)

            # Pixels de Floresta Natural serão somados, enquanto e o resto também
            pixel_floresta = pixel_simplified[0] + pixel_simplified[1]
            pixel_resto = sum(pixel_simplified[2:])

            # Criação do DataFrame e calculo do ângulo para o wedge da figura 02
            df = pd.DataFrame({'significado': significado_pixel_lista, 'value':pixel_simplified})
            df['angle'] = df['value']/df['value'].sum() * 2*np.pi

            self.source_02_03.data = dict(angle=df['angle'],
                                       color=Spectral10,
                                       significado=df['significado'])
            return pixel_floresta, pixel_resto

        if self.tabs.active == 3:
            # Dicionário e lista do significado do pixel do tif. Não foi inserido 255.
            significado_pixel = {3: 'Floresta Natural => Formação Florestal',
                                 4: 'Floesta Natural => Formação Savânica',
                                 9: 'Floresta Plantada',
                                 12: 'Formação Campestre/Outra formação não Florestal',
                                 15: 'Pastagem',
                                 19: 'Agricultura => Cultivo Anual e Perene',
                                 20: 'Agricultura => Cultivo Semi-Perene',
                                 24: 'Infraestrutura Urbana',
                                 25: 'Outra área não Vegetada',
                                 33: "Corpo d'água"}
            significado_pixel_lista = ['Floresta Natural (Formação Florestal)', 'Floesta Natural (Formação Savânica)',
                                       'Floresta Plantada', 'Formação Campestre', 'Pastagem', 'Agricultura (Cultivo Anual e Perene)',
                                       'Agricultura (Cultivo Semi-Perene)', 'Infraestrutura Urbana', 'Outra área não Vegetada', "Corpo d'água"]
            pixel_dict = dict(zip(unique, counts))

            # A partir do dicionário contiver counts será inserido na lista, caso não contiver 0 será aplicado.
            pixel_simplified = []
            for i in significado_pixel:
                try:
                    pixel_simplified.append(pixel_dict[i])
                except:
                    pixel_simplified.append(0)
            # print(pixel_simplified)
            # pixel_complete = dict(zip(significado_pixel_lista, pixel_simplified))
            # return pixel_simplified[0],pixel_simplified[1],pixel_simplified[2],pixel_simplified[3],pixel_simplified[4],pixel_simplified[5],pixel_simplified[6],pixel_simplified[7],pixel_simplified[8],pixel_simplified[9]
            return pixel_simplified

    def _button_download(self):
        if self.tabs.active == 3:
            datetime_start = dt.datetime.utcfromtimestamp(self.datetime_range.value[0]/1000)
            datetime_end = dt.datetime.utcfromtimestamp(self.datetime_range.value[1]/1000)

            inputs_to_k15 = self.df_ep.loc[(self.df_ep['TIMESTAMP']>=datetime_start) &
                                           (self.df_ep['TIMESTAMP']<=datetime_end), ['TIMESTAMP','date','time','u_rot', 'L', 'u*','v_var','wind_dir_compass']]
            print(inputs_to_k15)
            df_to_save = inputs_to_k15.copy()

            # classfication_list = []
            code03=[]
            code04=[]
            code09=[]
            code12=[]
            code15=[]
            code19=[]
            code20=[]
            code24=[]
            code25=[]
            code33=[]

            for index, row in inputs_to_k15.iterrows():
                print(row)
                try:
                    # Output para o footprint de Kljun et a. (2015)
                    # Para mudar a contribuição é necessário alterar o rs
                    out = self.k15_individual.output(zm=9,
                                                     umean=row['u_rot'],
                                                     h=1000,
                                                     ol=row['L'],
                                                     sigmav=row['v_var'],
                                                     ustar=row['u*'],
                                                     wind_dir=row['wind_dir_compass'],
                                                     rs=[0.3, 0.8], crop=False, fig=False)

                    # Criação do polígono do footprint 90% para Sirgas 2000 utm 23S com o ponto de referência a torre IAB3
                    poly = [(i+self.iab3_x_utm_sirgas, j+self.iab3_y_utm_sirgas) for i, j in zip(out[8][-1], out[9][-1])]
                    poly_shp = Polygon(poly)

                    # Mask utilzindo o arquivo tif e o polígono (ambos no mesmo sistema de coordenadas)
                    mask_teste = rasterio.mask.mask(self.raster, [poly_shp], crop=True, invert=False)

                    # Contabilização e contagem do Mask
                    unique, counts = np.unique(mask_teste[0], return_counts=True)

                    # Função para calcular estatísticas básicas do Mask
                    simplified_stats = self.stats_pixel(unique, counts)

                    # simplified_stats = stats.copy()
                    # print(type(simplified_stats))
                    # c03 = simplified_stats[0]
                    # classfication_list.append(simplified_stats)
                    # print(c03)
                    code03.append(simplified_stats[0])
                    # print(code03)

                    code04.append(simplified_stats[1])
                    code09.append(simplified_stats[2])
                    code12.append(simplified_stats[3])
                    code15.append(simplified_stats[4])
                    code19.append(simplified_stats[5])
                    code20.append(simplified_stats[6])
                    code24.append(simplified_stats[7])
                    code25.append(simplified_stats[8])
                    code33.append(simplified_stats[9])
                    # print(simplified_stats)
                except:
                    print('erro passou pro proximo')
                    code03.append('nan')
                    code04.append('nan')
                    code09.append('nan')
                    code12.append('nan')
                    code15.append('nan')
                    code19.append('nan')
                    code20.append('nan')
                    code24.append('nan')
                    code25.append('nan')
                    code33.append('nan')
                    # classfication_list.append(['nan','nan','nan','nan','nan','nan','nan','nan','nan','nan'])
            # df_to_save.join(pd.DataFrame())
            # df_to_save['number_of_pixel_classification'] = classfication_list
            # print(code03)
            df_to_save['code03'] = code03
            df_to_save['code04'] = code04
            df_to_save['code09'] = code09
            df_to_save['code12'] = code12
            df_to_save['code15'] = code15
            df_to_save['code19'] = code19
            df_to_save['code20'] = code20
            df_to_save['code24'] = code24
            df_to_save['code25'] = code25
            df_to_save['code33'] = code33
            #
            folder_to_save = pathlib.Path('{}'.format(self.path_download.value))

            file_name = 'classification_pixel_{}to{}.csv'.format(datetime_start.strftime('%Y-%m-%d-%H-%M'),datetime_end.strftime('%Y-%m-%d-%H-%M'))
            # print(file_name)
            df_to_save.to_csv(folder_to_save/file_name)
            # print(classfication_list)
            # print(len(classfication_list))

    def _button_update_heatmap(self):
        # self.source
        file = pathlib.Path('{}'.format(self.path_footprintStats_k15.value))
        self.df_footprintstats = pd.read_csv(file, na_values=[-9999,'nan'], parse_dates=['TIMESTAMP','date','time'])

        self.df_footprintstats['classification_percentage'] = (self.df_footprintstats['code03']+self.df_footprintstats['code04'])/(self.df_footprintstats['code03']+self.df_footprintstats['code04']+self.df_footprintstats['code09']+self.df_footprintstats['code12']+self.df_footprintstats['code15']+self.df_footprintstats['code19']+self.df_footprintstats['code20']+self.df_footprintstats['code24']+self.df_footprintstats['code25']+self.df_footprintstats['code33'])

        self.df_footprintstats['florest_s_percentage'] = (self.df_footprintstats['code04']/(self.df_footprintstats['code03']+self.df_footprintstats['code04']))

        self.df_footprintstats['resto_code'] = self.df_footprintstats['code09'] + self.df_footprintstats['code12'] +self.df_footprintstats['code15']+self.df_footprintstats['code19']+self.df_footprintstats['code20']+self.df_footprintstats['code24']+self.df_footprintstats['code25']+self.df_footprintstats['code33']
        self.df_footprintstats['pasto_percentage'] = (self.df_footprintstats['code15'])/(self.df_footprintstats['resto_code'])

        self.color_mapper_pixels.low = 0
        self.color_mapper_pixels.high = 1

        self.color_mapper_florest.low = 0
        self.color_mapper_florest.high = 1
        #
        self.source_04.data = dict(date=self.df_footprintstats['date'],
                                   time=self.df_footprintstats['time'],
                                   classification_pixel=self.df_footprintstats['classification_percentage'],
                                   wind_dir=self.df_footprintstats['wind_dir_compass'],
                                   florest_s_percentage=self.df_footprintstats['florest_s_percentage'],
                                   pasto_percentage=self.df_footprintstats['pasto_percentage'],
                                   code03=self.df_footprintstats['code03'],
                                   code04=self.df_footprintstats['code04'],
                                   code15=self.df_footprintstats['code15'],
                                   resto_code=self.df_footprintstats['resto_code'])


    def _teste1(self):
        print(dt.datetime.utcfromtimestamp(self.datetime_range.value[0]/1000), dt.datetime.utcfromtimestamp(self.datetime_range.value[1]/1000))


view_k15()
