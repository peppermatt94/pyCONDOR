# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 21:43:36 2021

@author: pepermatt94
"""
import pandas as pd
import numpy as np


class IMPS(object):
    def __init__(self, omega, H_prime, H_double_prime):
        self.omega = omega
        self.H_prime = H_prime
        self.H_double_prime = H_double_prime
        self.tau_vec = 2*np.pi/self.omega
        self.tau_fine  = np.logspace(np.log10(self.tau_vec.min())-0.5,np.log10(self.tau_vec.max())+0.5,10*self.omega.shape[0])

    @classmethod
    def from_file(cls, filename):
        data = np.loadtxt(filename, delimiter = " ",skiprows = 0)# encoding = "latin")
        omega = data[:,0]#2
        H_prime = data[:,1]# c'era un -
        H_double_prime = data[:,2]#1
        return cls(omega, H_prime, H_double_prime)
    @classmethod
    def from_array(cls, omega, H_prime, H_double_prime):
        return cls(omega, H_prime, H_double_prime)
