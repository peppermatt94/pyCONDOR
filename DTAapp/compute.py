import matplotlib.pylab as plt
import numpy as np
import os
import DTAapp.dtatools as DTA
from DTAapp.selection_range import range_Selection 
import cvxpy as cp
from scipy.interpolate import interp1d
def compute(IMPS_object, lambda_val):
    model_real = interp1d(IMPS_object.omega, IMPS_object.H_prime)
    model_double_prime = interp1d(IMPS_object.omega, IMPS_object.H_double_prime)

    omega_vec= np.logspace(np.log10(IMPS_object.omega[1]), np.log10(IMPS_object.omega[-1]), 40)
    real = model_real(omega_vec)
    imag = model_double_prime(omega_vec)

    #omega_vec = IMPS_object.omega
    tau_vec = 2*np.pi/omega_vec
    #real = IMPS_object.H_prime
    #imag = IMPS_object.H_double_prime
    mu= DTA.shape_factor_estimation(omega_vec, rbf_method = "Gaussian")
    real_matr= DTA.real_matrix(omega_vec, tau_vec, shape_factor = mu,rbf_method = "Gaussian" )
    imag_matr= DTA.imag_matrix(omega_vec, tau_vec, shape_factor = mu, rbf_method = "Gaussian")
    tich_matr= DTA.tichonov_matrix(tau_vec, shape_factor=mu, rbf_method = "Gaussian")  

    N_RL = 0 # N_RL length of resistence plus inductance
    A_re = np.zeros((omega_vec.size, tau_vec.size+N_RL))
    A_re[:,N_RL:] = real_matr
    A_re[:,0] = 1        
    A_im = np.zeros((omega_vec.size, tau_vec.size+N_RL))
    A_im[:,N_RL:] = imag_matr
    A_im[:,0] = omega_vec
    M =  np.zeros((omega_vec.size+N_RL, tau_vec.size+N_RL))
    M[N_RL:,N_RL:] = tich_matr

    Q, c = DTA.construct_convex_matrix(A_re, A_im, real, imag, lambda_val, M)
    N_out = c.shape[0]
    x = cp.Variable(shape = N_out, value = np.ones(N_out))
    h = np.zeros(N_out)
    I = np.identity(N_out)    
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x,Q) + c@x))#, [x>=h])
    prob.solve(verbose = True, eps_abs = 1E-10, eps_rel = 1E-10, sigma = 1.00e-08, 
               max_iter = 20000000, eps_prim_inf = 1E-5, eps_dual_inf = 1E-5)
    gamma = x.value
    DTA_vec, out = DTA.come_back_to_DTA(gamma[N_RL:], IMPS_object.tau_fine, tau_vec, mu, "Gaussian")
    mu_z_re = A_re@gamma
    mu_z_im = A_im@gamma
    fig = plt.figure()
    grid = fig.add_gridspec(2,1)
    fit = fig.add_subplot(grid[0,0])
    dta = fig.add_subplot(grid[1,0])
    fit.plot(real,imag, "o")
    fit.set_xlabel("Y'")
    fit.set_ylabel("Y''")
    fit.plot(mu_z_re, mu_z_im, "--")
    path1 =  os.path.join('DTAapp', 'img', 'fit.png')
    alls = "D:\\DTA\\DTAtools\\DTAapp\\static\\"
    path2 =  os.path.join('DTAapp', 'img', 'dta.png')
    dta.semilogx(out, DTA_vec)
    dta.set_xlabel("$\\tau (s)$")
    dta.set_ylabel("DTA"+"$(V^{-1})$")
    fig.suptitle("FIT")
    fig.savefig(alls+path1, dpi = 250)
    return path1, path2

def interpolate(IMPS_object):
    model_real = interp1d(IMPS_object.omega, IMPS_object.H_prime)
    model_double_prime = interp1d(IMPS_object.omega, IMPS_object.H_double_prime)
    omega_vec= np.logspace(np.log10(IMPS_object.omega[1]), np.log10(IMPS_object.omega[-1]), 40)
    real = model_real(omega_vec)
    imag = model_double_prime(omega_vec)
    return real, imag, omega_vec


def compute_bokeh(IMPS_object, lambda_val, voltage, output_dataframe1, output_dataframe2, degree):
    
    real, imag,omega_vec= interpolate(IMPS_object) 
    tau_vec = 2*np.pi/omega_vec
    mu= DTA.shape_factor_estimation(omega_vec, rbf_method = "Gaussian")
    real_matr= DTA.real_matrix(omega_vec, tau_vec, shape_factor = mu,rbf_method = "Gaussian" )
    imag_matr= DTA.imag_matrix(omega_vec, tau_vec, shape_factor = mu, rbf_method = "Gaussian")
    tich_matr= DTA.tichonov_matrix(tau_vec, mu, "Gaussian", degree)  
    
    N_RL = 0 # N_RL length of resistence plus inductance
    A_re = np.zeros((omega_vec.size, tau_vec.size+N_RL))
    A_re[:,N_RL:] = real_matr
    A_re[:,0] = 1        
    A_im = np.zeros((omega_vec.size, tau_vec.size+N_RL))
    A_im[:,N_RL:] = imag_matr
    A_im[:,0] = omega_vec
    M =  np.zeros((omega_vec.size+N_RL, tau_vec.size+N_RL))
    M[N_RL:,N_RL:] = tich_matr
    
    Q, c = DTA.construct_convex_matrix(A_re, A_im, real, imag, lambda_val, M)
    N_out = c.shape[0]
    x = cp.Variable(shape = N_out, value = np.ones(N_out))
    h = np.zeros(N_out)
    I = np.identity(N_out)    
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x,Q) + c@x))#, [x>=h])
    prob.solve(verbose = True, eps_abs = 1E-10, eps_rel = 1E-10, sigma = 1.00e-08, 
               max_iter = 20000000, eps_prim_inf = 1E-5, eps_dual_inf = 1E-5)
    gamma = x.value
    DTA_vec, out = DTA.come_back_to_DTA(gamma[N_RL:], IMPS_object.tau_fine, tau_vec, mu, "Gaussian")
    mu_z_re = A_re@gamma
    mu_z_im = A_im@gamma
    
    output_dataframe1[f"freq {voltage}"] = omega_vec
    output_dataframe1[f"real {voltage}"] = real
    output_dataframe1[f"imag {voltage}"] = imag
    output_dataframe1[f"fit_real {voltage}"] = mu_z_re
    output_dataframe1[f"fit_imag {voltage}"] = mu_z_im
    output_dataframe2[f"tau {voltage}"] = out
    output_dataframe2[f"DTA {voltage}"] = DTA_vec
    
    return output_dataframe1, output_dataframe2
