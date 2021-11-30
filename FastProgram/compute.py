import matplotlib.pylab as plt
import numpy as np
import os
import dtatools as DTA
import cvxpy as cp
from scipy.interpolate import interp1d
from equispaced_points import function_spaced as funspace
from trial_selection_range import range_Selection
def compute(IMPS_object,interpolation_point, lambda_val, degree, discretization_method, fit, dta, color, label, value_of_treshold_derivative):
    model_real = interp1d(IMPS_object.omega, IMPS_object.H_prime)
    model_double_prime = interp1d(IMPS_object.omega, IMPS_object.H_double_prime)

    omega_vec= np.logspace(np.log10(IMPS_object.omega[0]) , np.log10(IMPS_object.omega[-1]), 40)
    real = model_real(omega_vec)
    #omega_vec = range_Selection(real, omega_vec ,value_of_treshold_derivative)
   
    #real = model_real(omega_vec)
    imag = model_double_prime(omega_vec)
    tau_vec = 2*np.pi/omega_vec
    mu= DTA.shape_factor_estimation(omega_vec, rbf_method = "Gaussian") 
    real_matr= DTA.real_matrix(omega_vec, tau_vec, shape_factor = mu,rbf_method = discretization_method)
    imag_matr= DTA.imag_matrix(omega_vec, tau_vec, shape_factor = mu, rbf_method = discretization_method)
    tich_matr= DTA.tichonov_matrix(tau_vec, shape_factor=mu, rbf_method = discretization_method, degree = degree)  

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
    #x=np.arange(0,50,0.1)
   
    fit.plot(IMPS_object.H_prime,IMPS_object.H_double_prime, "o", color = color, label= label)
    fit.plot(mu_z_re, mu_z_im, "--", color = color)
    path1 =  os.path.join('DTAapp', 'img', 'fit.png')
    alls = "D:\\DTA\\DTAtools\\DTAtools\\static\\"
    path2 =  os.path.join('DTAapp', 'img', 'dta.png')
    dta.semilogx(out, DTA_vec, color = color, label = label)
    #dta.semilogx( 1/IMPS_object.omega , -IMPS_object.H_double_prime, "o", color = color)
    #fig.savefig(alls+path1, dpi = 250)
    
