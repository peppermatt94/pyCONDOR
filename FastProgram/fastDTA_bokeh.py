# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:08:33 2021

@author: pepermatt94
"""

from FILE_setup import setting, palette, Voltage_list
from compute_bokeh import compute
import sys
#sys.path.append("C:\\Users\pepermatt94\\OneDrive - Alma Mater Studiorum Università di Bologna\\Master_Thesis\\DTA")
#from imps_simulation import Y_sl
from dta_main import IMPS
import matplotlib.pylab as plt
import numpy as np
import matplotlib.colors as mcolors
file_path = "C:\\Users\\pepermatt94\\OneDrive - Alma Mater Studiorum Università di Bologna\\Master_Thesis\\DRT\\IMPS_WO3BiVO4\\"
file_path = file_path +"IMPS_WO3_BiVO4_blue_5mWamm.txt"
alb = "C:\\Users\\pepermatt94\\Desktop\\datiNuovi.txt"
from bokeh.plotting import ColumnDataSource, figure#, curdoc 
from bokeh.plotting import show as showPlot
from bokeh.io import show, curdoc
from bokeh.models import CustomJS, RadioButtonGroup
from bokeh.layouts import layout
from bokeh.models import HoverTool
import pandas as pd
class picked_data:
    def __init__(self, x,y, omega):
        self.x = x
        self.y=y
        self.omega = omega
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('click on points')
        self.finish = False
        self.point = np.array([])

        self.line, = self.ax.plot(imps.H_prime, imps.H_double_prime, 'o',
                    picker=True, pickradius=0.5)  # 5 points tolerance
        self.cidpress = self.fig.canvas.mpl_connect('pick_event', self)
      #  self.cidclose = self.fig.canvas.mpl_connect('close_event', self.close)
        #self.fig.show()
    def __call__(self, event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        left = (xlim[1]-xlim[0])/4
        up = 3*(ylim[1]-ylim[0])/4
        displacement = (ylim[1]-ylim[0])/8 
        ind = event.ind
        if self.point.shape[0] == 0:
            self.ax.text(left, up, f"start : {xdata}, {ydata}")
            self.point = np.append(self.point,(xdata[ind], ydata[ind]))
            print(self.point) 
        elif self.point.shape[0] == 2:
            self.ax.text(left, up, f"start : {xdata}, {ydata}")
            self.point = np.append(self.point,(xdata[ind], ydata[ind]))
            print(self.point)
        elif self.point.shape[0] >2:
            self.point = np.array([])
            print(self.point)
filename = sys.argv[1]
Voltages = Voltage_list(filename)
color_list_until_green = palette((0,1,0), (1,1,0), int(len(Voltages)/2))
color_list_until_yellow = palette((1,0.9,0), (1,0,0), int(len(Voltages)/2))
color_list = color_list_until_green + color_list_until_yellow
serie = np.arange(1,len(Voltages)*2+1, 1, dtype = int)
serie = serie.reshape(len(Voltages),2)
#freq_choose =pd.read_csv(filename, sep = "\t", skiprows = [1,2])
#freq = pd.to_numeric(freq_choose[freq_choose.columns[0]], errors = "coerce")   
#df = pd.DataFrame({"freq": freq})
df_data = {}
df_dta = {}
for step, dati in enumerate(serie):
    setting(dati[0],dati[1],filename)
    imps = IMPS.from_file("trainingData.txt")
    df_data, df_dta = compute(imps, 1, Voltages[step], df_data, df_dta)

dati_interp_fit = pd.DataFrame(df_data)
DTA_tau = pd.DataFrame(df_dta)
dati_interp_fit.to_csv("interp_fit.csv", index = False)
DTA_tau.to_csv("dta_tau.csv", index = False)


