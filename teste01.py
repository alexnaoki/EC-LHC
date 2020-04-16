import ipywidgets
import pandas as pd
import numpy as np
import pathlib
import datetime as dt
# import matplotlib.pyplot as plt
# from bqplot import pyplot as plt
import bqplot as bq

class teste:
    def __init__(self):
        self.button = ipywidgets.Button(description='Plot')
        self.button.on_click(self.button_plot)
        self.x_scale_01_04 = bq.OrdinalScale(domain=[(dt.datetime(2000,1,1) + dt.timedelta(minutes=i*30)).strftime('%H:%M') for i in range(48)])
        self.y_scale_01_05 = bq.LinearScale()

        self.x_axis_01_04 = bq.Axis(scale=self.x_scale_01_04, label='x',tick_rotate=270)
        self.y_axis_01_05 = bq.Axis(scale=self.y_scale_01_05, label='y', orientation='vertical')

        self.fig_01_04 = bq.Figure(marks=[], axes=[self.x_axis_01_04, self.y_axis_01_05], animation_duration=500)

        self.bar_01 = bq.Bars(scales={'x':self.x_scale_01_04, 'y':self.y_scale_01_05})
        self.fig_01_04.marks = [self.bar_01]
        self.bar_01.x = ['00:00']
        self.bar_01.y = [1]
        self.out = ipywidgets.Output()

        self.accordion = ipywidgets.Accordion()
        self.accordion.children = [ipywidgets.VBox([self.button, self.fig_01_04,self.out])]

        self.tab = ipywidgets.Tab([self.accordion])

        display(self.tab)

    def button_plot(self, *args):
        with self.out:
            print(self.bar_01)
            self.bar_01.x = ['00:00','01:00','03:30','16:30']
            self.bar_01.y = [1,2,3,50]
