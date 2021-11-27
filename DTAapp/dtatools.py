import pandas as pd
import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint, fsolve
import matplotlib.pylab as plt
from scipy.integrate import quad 
import cvxpy as cp
import matplotlib.colors as mcolors
from scipy.linalg import toeplitz
import sys
sys.path.append("C:\\Script\\PythonLibrary")
from LabLibrary import sci_notation_as_Benini_want


def real_matrix_element(tau_n, omega_m, shape_factor, rbf_method):
    rbf_type = {
            "Gaussian": lambda x : np.exp(-(shape_factor*x)**2)
            }
    rbf = rbf_type.get(rbf_method)
    #  Now i build the integral for A': see documentation eq 2.24
    integrand = lambda y: (1/(1+(tau_n*omega_m)**2*np.exp(2*y)))*rbf(y)
    value_with_error = quad(integrand, -50, 50)
    value = value_with_error[0]
    return value

def imag_matrix_element(tau_n, omega_m, shape_factor, rbf_method):
    rbf_type = {
            "Gaussian": lambda x : np.exp(-(shape_factor*x)**2)
            }
    rbf = rbf_type.get(rbf_method)
    #  Now i build the integral for A'': see documentation eq 2.25
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

# SINCE WE HAVE NOW A PROBLEM IN G(S) WE NEED TO COME BACK TO h(tau)
#def come_back_to_DTA(minimization_output, tau_vec, shape_factor, rbf_method):

def come_back_to_DTA(minimization_output, tau_map_vec, tau_vec, shape_factor, rbf_method):

    rbf_type = {
            "Gaussian": lambda x : np.exp(-(shape_factor*x)**2)
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
            "Gaussian": lambda x, tau : -shape_factor**2*(2*x- 2*np.log(tau))*np.exp(-shape_factor**2*(x - np.log(tau))**2) 
           }
    
    rbf_type2 = {
             "Gaussian": lambda x, tau: shape_factor**4*(2*x - 2*np.log(tau))**2*np.exp(-shape_factor**2*(x - np.log(tau))**2) - 2*shape_factor**2*np.exp(-shape_factor**2*(x - np.log(tau))**2)
             } 
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
                }
    rbf = rbf_switch.get(rbf_method)
    FWHM_coeff = 2*fsolve(rbf,1)
    delta = np.mean(np.diff(np.log(1/freq.reshape(freq.size))))
    shape_factor = coeff*FWHM_coeff/delta
    return shape_factor[0]

def construct_convex_matrix(A_re, A_im, b_re, b_im, lambda_value, M):
    """
    This function need to construct the matrix for the optimization.
    We choose the quadratic form optimization like those present in wan     
    and ciucci doi ... 
    """
    Q = 2*((A_re.T@A_re+A_im.T@A_im)+lambda_value*M)
    Q = (Q.T+Q)/2
    c = -2*(b_im.T@A_im+b_re.T@A_re)
    return Q,c

def tichonov_error(x, Q, c, b_re, b_im):
    return 0.5*(x.T@Q@x) + c.T@x ##+ b_re.T@b_re +b_im.T@b_im

if __name__ == "__main__":
    from IMPS_simulation import Y_sl
    omega_vec = np.logspace(6,-2, 40)
    lambda_val = 1
    C = 4e-6*1e-5/(4e-6+1e-5)
    R = 10
    tau = R*C
    tau_vec= 2*np.pi/omega_vec
    tau_fine  = np.logspace(np.log10(tau_vec.min())-0.5,np.log10(tau_vec.max())+0.5,10*omega_vec.shape[0])
    mu= shape_factor_estimation(omega_vec, rbf_method = "Gaussian")
    real_matr= real_matrix(omega_vec, tau_vec, shape_factor = mu,rbf_method = "Gaussian" )
    imag_matr= imag_matrix(omega_vec, tau_vec, shape_factor = mu, rbf_method = "Gaussian")
    tich_matr= tichonov_matrix(tau_vec, shape_factor=mu, rbf_method = "Gaussian")  

    N_RL = 1 # N_RL length of resistence plus inductance
    A_re = np.zeros((omega_vec.size, tau_vec.size+N_RL))
    A_re[:,N_RL:] = real_matr
    A_re[:,0] = 1        
    A_im = np.zeros((omega_vec.size, tau_vec.size+N_RL))
    A_im[:,N_RL:] = imag_matr
    A_im[:,0] = omega_vec
    M = np.zeros((omega_vec.size+N_RL, tau_vec.size+N_RL))
    M[N_RL:,N_RL:] = tich_matr

# HERE PLOT OF THE SIMULATION IN IMPS-simulation.py OF PETER WORK
    characteristic = plt.figure(figsize = (14,7))
    fit_DTA = plt.figure(figsize = (14,7))
    analysis = fit_DTA.add_gridspec(1, 2)
    fit = fit_DTA.add_subplot(analysis[0,0])
    DTA_plot = fit_DTA.add_subplot(analysis[0,1])
    legend = ["$k_{ct} = 0 Hz$","$k_{ct} = 10 Hz$", "$k_{ct} = 50 Hz$","$k_{ct} = 100 Hz$","$k_{ct} = 200 Hz$"]
    color_list = [color for color in mcolors.TABLEAU_COLORS.values()]
    characteristic_time = np.zeros(5)
    for list_number, k1 in enumerate([0,10,50,100,200]):
        real, imag = Y_sl(omega_vec, C= C,tau = tau,C_sc=4e-6 ,k_2 = 10, k_1=k1)
        fit.plot(real, -imag, "o", label = legend[list_number], c = color_list[list_number])
        Q, c = construct_convex_matrix(A_re, A_im, real, imag, lambda_val, M)
        N_out = c.shape[0]
        x = cp.Variable(shape = N_out, value = np.ones(N_out))
        h = np.zeros(N_out)
        I = np.identity(N_out)    
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x,Q) + c@x))#, [x>=h])
        prob.solve(verbose = True, eps_abs = 1E-10, eps_rel = 1E-10, sigma = 1.00e-08, 
                   max_iter = 20000000, eps_prim_inf = 1E-5, eps_dual_inf = 1E-5)

        gamma = x.value
        DTA, out = come_back_to_DTA(gamma[N_RL:], tau_fine, tau_vec, mu, "Gaussian")
        mu_z_re = A_re@gamma
        mu_z_im = A_im@gamma
        fit.plot(mu_z_re, -mu_z_im, "-", c = color_list[list_number])
        DTA_plot.semilogx(out, DTA, c = color_list[list_number],label = legend[list_number])
        maximum_search = np.abs(DTA[out>1e-3])
        max_position = np.where(np.abs(DTA) == max(maximum_search))[0][0]
        characteristic_time[list_number] = out[max_position]
    ax1 = DTA_plot.twiny()
    ax1.set_xscale("log")
    ax1.set_xlim(DTA_plot.get_xlim())
    ax1.set_xticks(characteristic_time)
    names = [sci_notation_as_Benini_want(x) for x in characteristic_time]
    ax1.set_xticklabels(names)
    ax1.tick_params(axis="x", labelsize=8, labelrotation=70, labelleft = True) #, labelcolor="turquoise")
    ax1.grid(True)
    
    for number, value in enumerate(characteristic_time):
        ax1.get_xticklabels()[number].set_color(color_list[number]) 
    
    fit.legend()
    fit.set_title("FIT IN NYQUIST PLOT")
    fit.set_xlabel("$Y'(\omega)$")
    fit.set_ylabel("$-Y''(\omega)$")

    DTA_plot.legend()
    DTA_plot.set_title("DTA SPECTRA")
    DTA_plot.set_xlabel("$\\tau$"+" (s)")
    DTA_plot.set_ylabel("DTA "+"$(V^{-1})$")
    fit_DTA.suptitle("DTA ANALISYS OF PETER' SIMULATION")
    fit_DTA.savefig("DTA_analysis.png", dpi=250, bbox_inches = "tight")
    fit_DTA.show()
