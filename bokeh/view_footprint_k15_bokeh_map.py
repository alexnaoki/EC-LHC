from bokeh.io import show, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column, row
from bokeh.models import Slider, ColumnDataSource, Button, Tabs, Panel, DateSlider, Range1d, Div, TextInput, Select, Panel
from bokeh.tile_providers import ESRI_IMAGERY, get_provider

import sys, pathlib
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
        self.a = FFP()

        self.iab3_x_utm_webMarcator = -5328976.90
        self.iab3_y_utm_webMarcator = -2532052.38

        self.tif_file = r'C:\Users\User\git\EC-LHC\iab3_site\IAB1_SIRGAS_23S.tif'
        self.raster = rasterio.open(self.tif_file)
        self.iab3_x_utm_sirgas = 203917.07880027
        self.iab3_y_utm_sirgas = 7545463.6805863

        self.tabs = Tabs(tabs=[self.tab_01()])


        curdoc().add_root(column(self.tabs))

    def tab_01(self):
        x_range = (-5332000, -5327000)
        y_range = (-2535000, -2530000)


        tile_provider = get_provider(ESRI_IMAGERY)

        self.source = ColumnDataSource(data=dict(xrs=[], yrs=[]))
        fig01 = figure(title='K15', plot_height=500, plot_width=500, x_range=x_range, y_range=y_range)
        fig01.add_tile(tile_provider)

        teste01 = fig01.circle([self.iab3_x_utm_webMarcator], [self.iab3_y_utm_webMarcator], color='red')
        mlines = fig01.multi_line(xs='xrs', ys='yrs', source=self.source, color='red')




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

        self.div_01 = Div(text='Sem dados', width=500)

        tab01 = Panel(child=column(self.slider_zm,
                                   self.slider_umean,
                                   self.slider_h,
                                   self.slider_ol,
                                   self.slider_sigmav,
                                   self.slider_ustar,
                                   self.slider_wind_dir,
                                   self.div_01,
                                   fig01), title='Inputs K15')
        return tab01

    def run_footprint(self):
        out = self.a.output(zm=self.slider_zm.value,
                            umean=self.slider_umean.value,
                            h=self.slider_h.value,
                            ol=self.slider_ol.value,
                            sigmav=self.slider_sigmav.value,
                            ustar=self.slider_ustar.value,
                            wind_dir=self.slider_wind_dir.value,
                            rs=[0.3, 0.9], crop=False, fig=False)

        # Sirgas 2000 utm 23S
        poly = [(i+self.iab3_x_utm_sirgas, j+self.iab3_y_utm_sirgas) for i, j in zip(out[8][-1], out[9][-1])]
        poly_shp = Polygon(poly)

        mask_teste =rasterio.mask.mask(self.raster, [poly_shp], crop=True, invert=False)
        unique, counts = np.unique(mask_teste[0], return_counts=True)
        simplified_stats = self.stats_pixel(unique, counts)
        self.div_01.text = '''
        <table border="1"><tbody><tr>
			<td>Floresta&nbsp;</td>
			<td>Outros</td>
		</tr><tr>
			<td>&nbsp;{}</td>
			<td>{}</td>
		</tr></tbody></table>'''.format(simplified_stats[0], simplified_stats[1])


        # Web Marcator
        x_webMarcator = list(np.array(out[8][-1]) + self.iab3_x_utm_webMarcator)
        y_webMarcator = list(np.array(out[9][-1]) + self.iab3_y_utm_webMarcator)
        # print(x_webMarcator)
        # print(out[8][-1])
        self.source.data = dict(xrs=[x_webMarcator], yrs=[y_webMarcator])
        # print(out[8])
        # self.source.data = dict(xrs=out[8], yrs=out[9])

    def stats_pixel(self, unique, counts):
        significado_pixel = {3: 'Floresta Natural => Formação Florestal',
                             4: 'Floesta Natural => Formação Savânica',
                             9: 'Floresta Plantada',
                             12: 'Formação Campestre/Outra formação não Florestal',
                             15: 'Pastagem',
                             19: 'Agricultura => Cultivo Anual e Perene',
                             20: 'Agricultura => Cultivo Semi-Perene',
                             24: 'Infraestrutura Urbana',
                             25: 'Outra área não Vegetada',
                             33: "Corpo d'água",
                             255: 'Fora do escopo'}
        pixel_dict = dict(zip(unique, counts))

        pixel_simplified = []
        for i in significado_pixel:
            try:
                pixel_simplified.append(pixel_dict[i])
            except:
                pixel_simplified.append(0)
        pixel_floresta = pixel_simplified[0] + pixel_simplified[1]
        pixel_resto = sum(pixel_simplified[2:-1])
        return pixel_floresta, pixel_resto



view_k15()
