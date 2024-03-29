from bokeh.io import show, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column, row
from bokeh.models import Slider, ColumnDataSource, Button, Tabs, Panel, DateSlider, Range1d, Div, TextInput, Select, Panel
from bokeh.tile_providers import ESRI_IMAGERY, get_provider
from bokeh.transform import cumsum
from bokeh.palettes import Spectral10

import sys, pathlib

# Verificar o path da função FFP_python_v1_4 relativo a este arquivo .py
sys.path.append(str(pathlib.Path(sys.path[0]).parent/'footprint'/'FFP_python_v1_4'))

from calc_footprint_FFP_adjusted01 import FFP
import numpy as np
import pandas as pd
import datetime as dt
from shapely.geometry.polygon import Polygon
import rasterio
import rasterio.mask

class view_k15:
    def __init__(self):
        '''
        Visualizador do Footprint de um intervalo de medição utilizando o método por Kljun et al. (2015).
        "bokeh serve --show [Full Path deste arquivo .py]"
        '''
        # Inicialização da Função
        self.a = FFP()

        # Coordenadas x, y da torre IAB3 para web Marcartor
        self.iab3_x_utm_webMarcator = -5328976.90
        self.iab3_y_utm_webMarcator = -2532052.38

        # Localização do arquivo tif retirado do mapbiomas no sistema de coordenadas Sirgas 2000 23S
        self.tif_file = r'C:\Users\User\git\EC-LHC\iab3_site\IAB1_SIRGAS_23S.tif'
        self.raster = rasterio.open(self.tif_file)

        # Coordenadas x, y da torre IAB3 para Sirgas 2000 23S
        self.iab3_x_utm_sirgas = 203917.07880027
        self.iab3_y_utm_sirgas = 7545463.6805863

        # Inicialização do tab 01
        self.tabs = Tabs(tabs=[self.tab_01()])

        # Gerar servidor para rodar o programa
        curdoc().add_root(column(self.tabs))

    def tab_01(self):
        # Range do x, y relativo ao web Mercator
        x_range = (-5332000, -5327000)
        y_range = (-2535000, -2530000)

        # Tile para display do mapa na figura
        tile_provider = get_provider(ESRI_IMAGERY)

        # Figura 01 - Mapa e Footprint
        self.source = ColumnDataSource(data=dict(xrs=[], yrs=[]))
        fig01 = figure(title='K15', plot_height=500, plot_width=500, x_range=x_range, y_range=y_range)
        fig01.add_tile(tile_provider)

        # Glyphs da localização da torre IAB3 e contribuição de 90% do footprint
        teste01 = fig01.circle([self.iab3_x_utm_webMarcator], [self.iab3_y_utm_webMarcator], color='red')
        mlines = fig01.multi_line(xs='xrs', ys='yrs', source=self.source, color='red', line_width=1)

        # Figura 02 - Categoria do tipo de vegetação pelo footprint
        self.source_02 = ColumnDataSource(data=dict(angle=[], color=[], significado=[]))
        fig02 = figure(title='Categoria', plot_height=300, plot_width=500, x_range=(-1.6,1.4), toolbar_location=None)
        wedge = fig02.annular_wedge(x=-1, y=0, inner_radius=0.3, outer_radius=0.45,
                                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                                    line_color='white', fill_color='color', legend_field='significado', source=self.source_02)
        fig02.axis.axis_label=None
        fig02.axis.visible=False
        fig02.grid.grid_line_color = None
        fig02.outline_line_color = None

        # Widgets para Análise de sensibilidade do método Kljun et al. (2015)
        self.slider_zm = Slider(title='zm', start=0, end=20, step=0.1, value=1)
        self.slider_umean = Slider(title='Umean', start=0.1, end=10, step=0.01, value=1)
        self.slider_h = Slider(title='h', start=100, end=5000, step=1, value=1000)
        self.slider_ol = Slider(title='ol', start=-1000, end=1000, step=0.01, value=15)
        self.slider_sigmav = Slider(title='sigmav', start=0, end=5, step=0.1, value=0.5)
        self.slider_ustar = Slider(title='ustar', start=0.01, end=5, step=0.01, value=1)
        self.slider_wind_dir = Slider(title='wind dir', start=0, end=359, step=0.1,value=0)
        sliders = [self.slider_zm, self.slider_umean, self.slider_h,self.slider_ol,self.slider_sigmav, self.slider_ustar, self.slider_wind_dir]
        for slider in sliders:
            slider.on_change('value_throttled', lambda attr, old, new: self.run_footprint())

        # Tabela para basic stats
        self.div_01 = Div(text='''
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

        # Tab 01
        tab01 = Panel(child=column(self.slider_zm,
                                   self.slider_umean,
                                   self.slider_h,
                                   self.slider_ol,
                                   self.slider_sigmav,
                                   self.slider_ustar,
                                   self.slider_wind_dir,
                                   row(fig01, column(self.div_01,
                                                     fig02))), title='Inputs K15')
        return tab01

    def run_footprint(self):
        # Output para o footprint de Kljun et al. (2015)
        out = self.a.output(zm=self.slider_zm.value,
                            umean=self.slider_umean.value,
                            h=self.slider_h.value,
                            ol=self.slider_ol.value,
                            sigmav=self.slider_sigmav.value,
                            ustar=self.slider_ustar.value,
                            wind_dir=self.slider_wind_dir.value,
                            rs=[0.3, 0.9], crop=False, fig=False)

        # Criação do polígono do footprint 90% para Sirgas 2000 utm 23S com o ponto de referência a torre IAB3
        poly = [(i+self.iab3_x_utm_sirgas, j+self.iab3_y_utm_sirgas) for i, j in zip(out[8][-1], out[9][-1])]
        poly_shp = Polygon(poly)

        # Mask utilizando o arquivo tif e o polígono (ambos no mesmo sistema de coordenadas)
        mask_teste = rasterio.mask.mask(self.raster, [poly_shp], crop=True, invert=False)

        # Contabilização e contagem do Mask
        unique, counts = np.unique(mask_teste[0], return_counts=True)

        # Função para calcular estatísticas básicas do Mask
        simplified_stats = self.stats_pixel(unique, counts)

        # Atualização da tabela com os dados do basic stats
        self.div_01.text = '''
        <div class="header">
        <h1>Basic Stats</h1><table border="1"><tbody><tr>
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
        x_webMarcator = list(np.array(out[8][-1]) + self.iab3_x_utm_webMarcator)
        y_webMarcator = list(np.array(out[9][-1]) + self.iab3_y_utm_webMarcator)

        # Atualização do source
        self.source.data = dict(xrs=[x_webMarcator], yrs=[y_webMarcator])

    def stats_pixel(self, unique, counts):
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

        self.source_02.data = dict(angle=df['angle'],
                                   color=Spectral10,
                                   significado=df['significado'])

        return pixel_floresta, pixel_resto



view_k15()
