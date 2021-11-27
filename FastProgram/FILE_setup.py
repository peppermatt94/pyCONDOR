import numpy as np
import pandas as pd
import sys

def setting(H_prime, H_double_prime, filename):
    a = pd.read_csv(filename, sep = "\t", skiprows = [1,2])
    final = np.c_[pd.to_numeric(a[a.columns[0]],errors='coerce') , pd.to_numeric(a[a.columns[H_prime]],errors='coerce') , pd.to_numeric(a[a.columns[H_double_prime]],errors='coerce')]
    np.savetxt("trainingData.txt", final)
   
def Voltage_list(filename) : 
    Voltages = pd.read_csv(filename, delimiter = "\t", skiprows = 2, nrows=1)
    list_of_voltages = [2*i for i in range(int((len(Voltages.columns)-1)/2))]
    list_of_voltages.pop(0)
    Voltages = Voltages.columns[list_of_voltages]
    return Voltages


def palette(starting_tuple, final_tuple, nstep):
    increment = [starting_tuple[i]-final_tuple[i] for i, number in enumerate(starting_tuple)]
    palette = []
    for n in range(nstep):
        R = starting_tuple[0] - n*(increment[0]/nstep) 
        G = starting_tuple[1] - n*(increment[1]/nstep)
        B = starting_tuple[2] - n*(increment[2]/nstep)
        palette.append((R,G,B))
        
    return palette
