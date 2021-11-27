import pandas as pd
import numpy as np

def range_Selection(manifold_real, omega, value_of_treshold_derivative):
    omega= np.log10(omega)
    manifold_real_previous = manifold_real[:-1]
    manifold_real_next = np.delete(manifold_real,0)
    omega_previous = omega[:-1]
    omega_next = np.delete(omega,0)
    dxdy = (manifold_real_previous-manifold_real_next)/(omega_next-omega_previous)
    region_sel = []
    regions = np.logical_or(dxdy>value_of_treshold_derivative, dxdy<-0.006)
    low_region = omega[1:][regions][0] 
    omega_sel = np.array(omega[1:], dtype = 'object')
    omega_sel[regions] = 0
    deleting_index= []
    for i, value in enumerate(omega_sel):
        if value == 0 and omega_sel[i-1]!=0 and omega_sel[i+1]!=0:
            deleting_index.append(i)
    omega_sel = np.delete(omega_sel, deleting_index)
    omega_sel = np.split(omega_sel, np.where(omega_sel == 0)[0][1:])
    deleting_index = []
    for i, value in enumerate(omega_sel):
        if len(value)==1:
            deleting_index.append(i)
    omega_sel = np.delete(omega_sel, deleting_index)
    for i, value in enumerate(omega_sel):
        if i==0:
            region_sel.append([value[0],value[-2]])
        else:
            region_sel.append([value[1], value[-1]])
    interpolation_point = 40
    weight_of_extreme = 0.08

    weight_of_middle = 1-(0.08*len(region_sel))
    omega_vec = np.array([])
    """
    for i, value in enumerate(region_sel):
        omega_vec = np.append(omega_vec, np.logspace(value[0], value[1], int(interpolation_point*weight_of_extreme)))
        if i+1 < len(region_sel):
            omega_vec = np.append(omega_vec, np.logspace(value[1],region_sel[i+1][0], int(interpolation_point*weight_of_middle)))
    breakpoint()
    """
    omega_vec = np.append(omega_vec, np.logspace(region_sel[0][0], region_sel[0][1], int(interpolation_point*weight_of_extreme)))
    omega_vec = np.append(omega_vec, np.logspace(region_sel[0][1], omega[-1], int(interpolation_point*weight_of_middle))) 
    """
    omega_vec = np.append(omega_vec, np.logspace(region_sel[0][0], region_sel[0][1], int(interpolation_point*weight_of_extreme*2)))
    omega_vec = np.append(omega_vec, np.logspace(region_sel[0][1], region_sel[1][0], int(interpolation_point*weight_of_middle)))
    
    omega_vec = np.append(omega_vec, np.logspace(region_sel[1][0], region_sel[1][1],int(interpolation_point*weight_of_extreme)))
    #omega_vec = np.append(omega_vec, np.logspace(region_sel[1][1], omega[-1], int(interpolation_point*weight_of_middle/2.5)))
    #breakpoint()
    #breakpoint()
    """
    return omega_vec

