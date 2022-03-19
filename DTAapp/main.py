# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:08:33 2021

@author: pepermatt94
"""

from .FILE_setup import setting, palette, Voltage_list
from .compute import compute_bokeh, interpolate
import sys
from .DTA_main import IMPS
import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
from django.conf import settings

def main(file, rbf_method, lambda_val, degree_of_thickonov, n_interpolation_point, col_selection):
    rbf = {"1": "Gaussian",
           "2": "Inverse Quadric",
           "3": "Inverse Quadratic"
            }
    rbf_method = rbf.get(rbf_method) 
    Voltages = Voltage_list(file)
    selected_voltages = list(map(lambda x:int(x), col_selection.split(":")))
    if len(selected_voltages)==2:
        Voltages_list = Voltages[selected_voltages[0]:selected_voltages[1]]
    elif len(selected_voltages) == 1:
        if selected_voltages[0] == -1:
           Voltages_list = Voltages 
        else:
            Voltages_list = [Voltages[selected_voltages[0]]]
    
    color_list_until_green = palette((0,1,0), (1,1,0), int(len(Voltages)/2))
    color_list_until_yellow = palette((1,0.9,0), (1,0,0), int(len(Voltages)/2))
    color_list = color_list_until_green + color_list_until_yellow
    serie = np.arange(1,len(Voltages)*2+1, 1, dtype = int)
    serie = serie.reshape(len(Voltages),2)
    df_data = {}
    df_dta = {}
    for step, dati in enumerate(serie):
        if Voltages[step] in Voltages_list:
            setting(dati[0],dati[1],file)
            imps = IMPS.from_file("trainingData.txt")
            df_data, df_dta = compute_bokeh(imps, lambda_val, Voltages[step],rbf_method, df_data, df_dta, degree_of_thickonov)
    dati_interp_fit = pd.DataFrame(df_data)
    DTA_tau = pd.DataFrame(df_dta)
    dati_interp_fit.to_csv("interp_fit.csv", index = False)
    DTA_tau.to_csv("dta_tau.csv", index = False)
    return dati_interp_fit, DTA_tau

def interpolate_only(file, value_of_treshold_derivative, interpolation_type, n_interpolation_point, col_selection):
    
    Voltages = Voltage_list(file)
    selected_voltages = list(map(lambda x:int(x), col_selection.split(":")))
    if len(selected_voltages)==2:
        Voltages_list = Voltages[selected_voltages[0]:selected_voltages[1]]
    elif len(selected_voltages) == 1:
        if selected_voltages[0] == -1:
           Voltage_list =Voltages
        else:
            Voltages_list = [Voltages[selected_voltages[0]]]
    
    color_list_until_green = palette((0,1,0), (1,1,0), int(len(Voltages)/2))
    color_list_until_yellow = palette((1,0.9,0), (1,0,0), int(len(Voltages)/2))
    color_list = color_list_until_green + color_list_until_yellow
    serie = np.arange(1,len(Voltages)*2+1, 1, dtype = int)
    serie = serie.reshape(len(Voltages),2)
    for step, dati in enumerate(serie):
        if Voltages[step] in Voltages_list:
            setting(dati[0],dati[1],file)
            imps = IMPS.from_file("trainingData.txt")
            real, imag , omega_vec, dxdy= interpolate(imps, value_of_treshold_derivative, interpolation_type)
    
    return real ,imag, omega_vec, dxdy

