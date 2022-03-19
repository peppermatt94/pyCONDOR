import pandas as pd
import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint, fsolve
import matplotlib.pylab as plt
from scipy.integrate import quad 
import cvxpy as cp
import matplotlib.colors as mcolors
import sys

def real_matrix_element(tau_n, omega_m, shape_factor, rbf_method):
    rbf_type = {
            "Gaussian": lambda x : np.exp(-(shape_factor*x)**2),
            'Inverse Quadratic': lambda x: 1/(1+(shape_factor*x)**2),
            'Inverse Quadric': lambda x: 1/np.sqrt(1+(shape_factor*x)**2)
                }
    rbf = rbf_type.get(rbf_method)
    #  Now i build the integral for the ELEMENTS A': see master thesis eq. 2.20
    integrand = lambda y: (1/(1+(tau_n*omega_m)**2*np.exp(2*y)))*rbf(y)
    value_with_error = quad(integrand, -50, 50)
    value = value_with_error[0]
    return value

def imag_matrix_element(tau_n, omega_m, shape_factor, rbf_method):
    rbf_type = {
            "Gaussian": lambda x : np.exp(-(shape_factor*x)**2),
            #Not use: "C2 matern": lambda x = np.exp(-np.abs((shape_factor*x))*(1+shape_factor*x)
            'Inverse Quadratic': lambda x: 1/(1+(shape_factor*x)**2),
            'Inverse Quadric': lambda x: 1/np.sqrt(1+(shape_factor*x)**2)
            }
    rbf = rbf_type.get(rbf_method)
    #  Now i build the integral for the ELEMENTS of A'': see master thesis eq 2.20
    integrand = lambda y: (omega_m*tau_n*np.exp(y)/(1+(tau_n*omega_m)**2*np.exp(2*y)))*rbf(y)
    value_with_error = quad(integrand, -50, 50)
    value = value_with_error[0]
    return value

def real_matrix(omega, tau, rbf_method, shape_factor):
    n_col= tau.size
    n_row =omega.size
    A_real = np.zeros((n_row, n_col))
    for column, tau_value in enumerate(tau):
        for row, omega_value in enumerate(omega):
            A_real[row][column] = real_matrix_element(tau_value, omega_value, shape_factor, rbf_method)
    return A_real

def imag_matrix(omega, tau, rbf_method, shape_factor):
    n_col= tau.size
    n_row =omega.size
    A_imag = np.zeros((n_row, n_col))
    for column, tau_value in enumerate(tau):
        for row, omega_value in enumerate(omega):
            A_imag[row][column] = -imag_matrix_element(tau_value, omega_value, shape_factor, rbf_method)
    return A_imag

# SINCE WE HAVE NOW A PROBLEM IN G(S) WE NEED TO COME BACK TO h(tau). see master thesis eq. 2.16

def come_back_to_DTA(minimization_output, tau_map_vec, tau_vec, shape_factor, rbf_method):
    rbf_type = {
            "Gaussian": lambda x : np.exp(-(shape_factor*x)**2),
            'Inverse Quadratic': lambda x: 1/(1+(shape_factor*x)**2),
            'Inverse Quadric': lambda x: 1/np.sqrt(1+(shape_factor*x)**2),

            }
    rbf = rbf_type.get(rbf_method)
    DTA = np.zeros(tau_map_vec.size)
    B = np.zeros((tau_map_vec.size, tau_vec.size))        
    for p in range(0, tau_map_vec.size):
        for q in range(0, tau_vec.size):
            delta_log_tau = np.log(tau_map_vec[p])-np.log(tau_vec[q])
            B[p,q] = rbf(delta_log_tau)       
    DTA = B@minimization_output
    out_tau = tau_map_vec
    return DTA, out_tau

def tichonov_matrix_element(tau_n, tau_m, shape_factor, rbf_method, degree):
    rbf_type1 ={
            "Gaussian": lambda x, tau : -shape_factor**2*(2*x- 2*np.log(tau))*np.exp(-shape_factor**2*(x - np.log(tau))**2),
            'Inverse Quadratic': lambda x,tau: -shape_factor**2*(2*x - 2*np.log(tau))/(shape_factor**2*(x - np.log(tau))**2 + 1)**2,
            'Inverse Quadric': lambda x, tau: -shape_factor**2*(2*x - 2*np.log(tau))/(2*(shape_factor**2*(x - np.log(tau))**2 + 1)**(3/2))
            }
    
    rbf_type2 = {
             "Gaussian": lambda x, tau: shape_factor**4*(2*x - 2*np.log(tau))**2*np.exp(-shape_factor**2*(x - np.log(tau))**2) - 2*shape_factor**2*np.exp(-shape_factor**2*(x - np.log(tau))**2),
             'Inverse Quadratic': lambda x,tau: 2*shape_factor**4*(2*x - 2*np.log(tau))**2/(shape_factor**2*(x - np.log(tau))**2 + 1)**3 - 2*shape_factor**2/(shape_factor**2*(x - np.log(tau))**2 + 1)**2,
             'Inverse Quadric': lambda x, tau:3*shape_factor**4*(2*x - 2*np.log(tau))**2/(4*(shape_factor**2*(x - np.log(tau))**2 + 1)**(5/2)) - shape_factor**2/(shape_factor**2*(x - np.log(tau))**2 + 1)**(3/2) 
             } 
    # Now we construct M matrix: see master thesi eq. 2.28
    if degree == 1:
        rbf  = rbf_type1.get(rbf_method)
    elif degree == 2:
        rbf=rbf_type2.get(rbf_method)

    #  Now i build the integral for A': see documentation eq 2.24
    integrand = lambda y: rbf(y, tau_n)*rbf(y,tau_m)
    value_with_error = quad(integrand, -50, 50)
    value = value_with_error[0]
    return value


def tichonov_matrix(tau, shape_factor, rbf_method, degree):
    n_col= tau.size
    n_row =tau.size
    M = np.zeros((n_row, n_col))
    for column, tau_n in enumerate(tau):
        for row, tau_m in enumerate(tau):
            M[row][column] = tichonov_matrix_element(tau_n, tau_m, shape_factor, rbf_method, degree)
    return M

def shape_factor_estimation(omega_vec, rbf_method, coeff = 0.5): 
    freq = omega_vec/(2*np.pi)
    rbf_switch = {
                'Gaussian': lambda x: np.exp(-(x)**2)-0.5,
                'Inverse Quadratic': lambda x: 1/(1+(x)**2)-0.5,
                'Inverse Quadric': lambda x: 1/np.sqrt(1+(x)**2)-0.5
                }
    # We determines the shape factor. See master thesis eq. 2.22
    rbf = rbf_switch.get(rbf_method)
    FWHM_coeff = 2*fsolve(rbf,1)
    delta = np.mean(np.diff(np.log(1/freq.reshape(freq.size))))
    shape_factor = coeff*FWHM_coeff/delta
    return shape_factor[0]

def construct_convex_matrix(A_re, A_im, b_re, b_im, lambda_value, M):
    """
    This function need to construct the matrix for the optimization.
    We choose the quadratic form optimization like those present in wan     
    and ciucci. See master thesis eq. 2.33 
    """
    Q = 2*((A_re.T@A_re+A_im.T@A_im)+lambda_value*M)
    Q = (Q.T+Q)/2
    c = -2*(b_im.T@A_im+b_re.T@A_re)
    return Q,c

def tichonov_error(x, Q, c, b_re, b_im):
    return 0.5*(x.T@Q@x) + c.T@x ##+ b_re.T@b_re +b_im.T@b_im


