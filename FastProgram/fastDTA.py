# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:08:33 2021

@author: pepermatt94
"""

from FILE_setup import setting, palette, Voltage_list
from compute import compute
import sys
#sys.path.append("C:\\Users\pepermatt94\\OneDrive - Alma Mater Studiorum Università di Bologna\\Master_Thesis\\DTA")
#from imps_simulation import Y_sl
from dta_main import IMPS
import matplotlib.pylab as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

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
                    picker=True, pickradius=2)  # 5 points tolerance
        self.cidpress = self.fig.canvas.mpl_connect('pick_event', self)
        

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



fig = plt.figure(figsize=(14,7))
grid = fig.add_gridspec(1,2)
fit = fig.add_subplot(grid[0,0])
dta = fig.add_subplot(grid[0,1])
filename = sys.argv[1]
selected_voltages = list(map(lambda x:int(x), sys.argv[2].split(":")))
Voltages = Voltage_list(filename)
if len(selected_voltages)==2:
    Voltages_list = Voltages[selected_voltages[0]:selected_voltages[1]]
elif len(selected_voltages) == 1:
    if selected_voltages[0] == -1:
        Voltages_list = Voltages
    else:
        Voltages_list = Voltages[selected_voltages[0]]

#color_list_until_green = palette((0,1,0), (1,1,0), int(len(Voltages)+1/2))
#color_list_until_yellow = palette((1,0.9,0), (1,0,0), int(len(Voltages)+1/2))
#color_list = color_list_until_green + color_list_until_yellow
#color_list = palette((1,0,0), (1,1,0), int(len(Voltages)))
color_list = sns.color_palette(None, len(Voltages))

serie = np.arange(1,len(Voltages)*2+1, 1, dtype = int)
serie = serie.reshape(len(Voltages),2)
for step, dati in enumerate(serie):
    if Voltages[step] in Voltages_list:
        setting(dati[0],dati[1],filename)
        imps = IMPS.from_file("trainingData.txt")
        if sys.argv[4] != "silent":
            fig_selection = picked_data(imps.H_prime, imps.H_double_prime, imps.omega)
            plt.show()
            low = np.where(imps.H_prime == fig_selection.point[0])[0][0]
            high = np.where(imps.H_prime == fig_selection.point[2])[0][0] 
            omega = imps.omega[low:high]
            H_prime = imps.H_prime[low:high]
            H_double_prime = imps.H_double_prime[low:high]
            imps = IMPS.from_array(omega, H_prime, H_double_prime)
        #fig = plt.figure(figsize=(14,7))
        #grid = fig.add_gridspec(1,2)
        #fit = fig.add_subplot(grid[0,0])
        #dta = fig.add_subplot(grid[0,1])
        compute(imps,int(sys.argv[3]),  0.001,2, "Gaussian",  fit, dta, color_list[step], Voltages[step], 6e-3)
fit.legend()
fit.set_xlabel("Y'")
fit.set_ylabel("Y''")
dta.set_xlabel("$\\tau (s)$")
dta.set_ylabel("DTA"+"$(V^{-1})$")
filename = filename.replace("cat", "CoFe-Bp")
fig.suptitle(filename[:-4])
plt.savefig(filename[:-4] + ".png", dpi=250)
plt.close()
