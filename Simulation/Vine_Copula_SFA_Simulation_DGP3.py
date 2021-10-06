# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:44:28 2021

@author: Robert James
"""
import matlab.engine
import numpy as np 
import pandas as pd
import multiprocessing
import os
import random

from scipy import stats
from tqdm import tqdm 

def AppEstimate_QMLE(y, x1, x2, z1, z2, eng):
    #Define initial values for maximization of the log-likelihood
    initial_alpha = 0.5
    initial_beta1 = 0.5
    initial_beta2 = 0.5
    initial_sigma2u = 0.5
    initial_sigma2v = 0.5
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigma2w1 = 1
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_sigma2w2 = 1
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_gamma = 1
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_z1 = matlab.double(z1.tolist())
    matlab_z2 = matlab.double(z2.tolist())
    
    theta, sterr, u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_QMLE(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_theta0, nargout = 4)
    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.array(u_hat)
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2)
    full_theta = np.concatenate([theta, np.array([np.nan, np.nan, np.nan])])
    return full_theta, theta, sterr, u_hat, u_hat_conditional_w1w2

def AppEstimate_Gaussian_FMLE(y, x1, x2, z1, z2, w1, w2, us, random_seed, eng):
    #Define initial values for maximization of the log-likelihood
    initial_alpha = 0.5
    initial_beta1 = 0.5
    initial_beta2 = 0.5
    initial_sigma2u = 1
    initial_sigma2v = 1
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigma2w1 = 1
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_sigma2w2 = 1
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_rho = 0.2
    initial_gamma = 1
    initial_pw1w2 = stats.kendalltau(stats.norm.cdf(w1, loc = 0 , scale = np.sqrt(initial_sigma2w1)), stats.norm.cdf(w2, loc = 0 , scale = np.sqrt(initial_sigma2w2)))[0]
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_rho, initial_pw1w2])

    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_z1 = matlab.double(z1.tolist())
    matlab_z2 = matlab.double(z2.tolist())
    matlab_us = matlab.double(us.tolist())

    tmp_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_Gaussian(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
    tmp_theta = np.array(tmp_theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    theta = tmp_theta[:-2] #Dont need corr(w1,w2) and rho
    u_hat = np.array(u_hat)
    non_par_u_hat = np.array(non_par_u_hat)
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2) 
    full_theta = np.concatenate([theta, np.array([np.nan, np.nan, tmp_theta[-1]])])
    return full_theta, theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2

def AppEstimate_APS3B(y, x1, x2, z1, z2, w1, w2, us, random_seed, eng, true_theta12, true_theta13):
    #Define initial values for maximization of the log-likelihood
    initial_theta12_dict = {0.3:0.1, 0.45:0.3, 0.8:0.4, 0.9:0.4}
    initial_theta13_dict = {0.1:0.05, 0.45:0.3, 0.7:0.4, -0.7:-0.4}
    initial_alpha = 0.5
    initial_beta1 = 0.5
    initial_beta2 = 0.5
    initial_sigma2u = 1
    initial_sigma2v = 1
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigma2w1 = 1
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_sigma2w2 = 1
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_gamma = 1
    initial_theta12 = initial_theta12_dict[true_theta12]
    initial_theta13 = initial_theta13_dict[true_theta13]
    initial_pw1w2 = stats.kendalltau(stats.norm.cdf(w1, loc = 0 , scale = np.sqrt(initial_sigma2w1)), stats.norm.cdf(w2, loc = 0 , scale = np.sqrt(initial_sigma2w2)))[0]
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_theta12, initial_theta13, initial_pw1w2])

    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_z1 = matlab.double(z1.tolist())
    matlab_z2 = matlab.double(z2.tolist())
    matlab_us = matlab.double(us.tolist())

    try: #Estimate the copula parameters
        full_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_APS3B(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
    except matlab.engine.MatlabExecutionError: #If we have an error try other random starting values for theta12, theta13
        estimation_exit_bool = False
        i = 0
        while estimation_exit_bool == False:
            if i > random_param_trial_runs:
                estimation_exit_bool = True
            initial_theta12 = random.uniform(-0.99, 0.99)
            initial_theta13 = random.uniform(-0.99, 0.99)
            theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_theta12, initial_theta13, initial_pw1w2])
            matlab_theta0 = matlab.double(theta0.tolist())
            matlab_theta0.reshape((len(theta0),1))
            try:
                full_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_APS3B(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
                estimation_exit_bool = True
            except matlab.engine.MatlabExecutionError:
                i += 1
                continue

    full_theta = np.array(full_theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    theta = full_theta[:-3]
    u_hat = np.array(u_hat)
    non_par_u_hat = np.array(non_par_u_hat)
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2)    
    return full_theta, theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2

def AppEstimate_APS3A(y, x1, x2, z1, z2, w1, w2, us, random_seed, eng, true_theta12, true_theta13):
    #Define initial values for maximization of the log-likelihood
    initial_theta12_dict = {0.3:0.1, 0.45:0.3, 0.8:0.4, 0.9:0.4}
    initial_theta13_dict = {0.1:0.05, 0.45:0.3, 0.7:0.4, -0.7:-0.4}
    initial_alpha = 0.5
    initial_beta1 = 0.5
    initial_beta2 = 0.5
    initial_sigma2u = 1
    initial_sigma2v = 1
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigma2w1 = 1
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_sigma2w2 = 1
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_gamma = 1
    initial_theta12 = initial_theta12_dict[true_theta12]
    initial_theta13 = initial_theta13_dict[true_theta13]
    initial_pw1w2 = stats.kendalltau(stats.norm.cdf(w1, loc = 0 , scale = np.sqrt(initial_sigma2w1)), stats.norm.cdf(w2, loc = 0 , scale = np.sqrt(initial_sigma2w2)))[0]
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_theta12, initial_theta13, initial_pw1w2])

    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_z1 = matlab.double(z1.tolist())
    matlab_z2 = matlab.double(z2.tolist())
    matlab_us = matlab.double(us.tolist())

    try: #Estimate the copula parameters
        full_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_APS3A(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
    except matlab.engine.MatlabExecutionError: #If we have an error try other random starting values for theta12, theta13
        estimation_exit_bool = False
        i = 0
        while estimation_exit_bool == False:
            if i > random_param_trial_runs:
                estimation_exit_bool = True
            initial_theta12 = random.uniform(-0.49, 0.49)
            initial_theta13 = random.uniform(-0.49, 0.49)
            theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_theta12, initial_theta13, initial_pw1w2])
            matlab_theta0 = matlab.double(theta0.tolist())
            matlab_theta0.reshape((len(theta0),1))
            try:
                full_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_APS3A(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
                estimation_exit_bool = True
            except matlab.engine.MatlabExecutionError:
                i += 1
                continue

    full_theta = np.array(full_theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    theta = full_theta[:-3]
    u_hat = np.array(u_hat)
    non_par_u_hat = np.array(non_par_u_hat)
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2)    
    return full_theta, theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2

def AppEstimate_Vine_APS3A(y, x1, x2, z1, z2, w1, w2, us, random_seed, eng, true_theta12, true_theta13):
    #Define initial values for maximization of the log-likelihood
    initial_theta12_dict = {0.3:0.1, 0.45:0.3, 0.8:0.4, 0.9:0.4}
    initial_theta13_dict = {0.1:0.05, 0.45:0.3, 0.7:0.4, -0.7:-0.4}
    initial_alpha = 0.5
    initial_beta1 = 0.5
    initial_beta2 = 0.5
    initial_sigma2u = 1
    initial_sigma2v = 1
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigma2w1 = 1
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_sigma2w2 = 1
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_gamma = 1
    initial_theta12 = initial_theta12_dict[true_theta12]
    initial_theta13 = initial_theta13_dict[true_theta13]
#    b = 1
#    a = -1
#    initial_arctanh_theta12 = np.arctanh((initial_theta12 - (b + a)/2)/((b - a)/2))
#    initial_arctanh_theta13 = np.arctanh((initial_theta13 - (b + a)/2)/((b - a)/2))
    
    #Conditional CDF transforms for allocative inefficiency correlation starting value
    CDF_simulated_us = 2*(stats.norm.cdf(np.sqrt(initial_sigma2u)*us[0,:], 0, np.sqrt(initial_sigma2u)) -0.5).ravel() #use the first random sample of u in the intial guess for pw1w2
    CDF_w1 = stats.norm.cdf(w1, loc = 0 , scale = np.sqrt(initial_sigma2w1)).ravel()
    CDF_w2 = stats.norm.cdf(w2, loc = 0 , scale = np.sqrt(initial_sigma2w2)).ravel()
#    APS3A_CDF_w1_conditional_u = CDF_w1 + (initial_theta12*CDF_w1)*(1 - CDF_w1)*(2*(6*CDF_simulated_us - 6*CDF_simulated_us**2 - 1))
#    APS3A_CDF_w2_conditional_u = CDF_w2 + (initial_theta13*CDF_w2)*(1 - CDF_w2)*(2*(6*CDF_simulated_us - 6*CDF_simulated_us**2 - 1))
    APS3A_CDF_w1_conditional_u = CDF_w1 + initial_theta12*CDF_simulated_us*CDF_w1*(4*CDF_w1**2 - 6*CDF_w1 + 2) + initial_theta12*CDF_w1*(CDF_simulated_us - 1)*(4*CDF_w1**2 - 6*CDF_w1 + 2)
    APS3A_CDF_w2_conditional_u = CDF_w2 + initial_theta13*CDF_simulated_us*CDF_w2*(4*CDF_w2**2 - 6*CDF_w2 + 2) + initial_theta13*CDF_w2*(CDF_simulated_us - 1)*(4*CDF_w2**2 - 6*CDF_w2 + 2)
    initial_pw1w2 = stats.kendalltau(APS3A_CDF_w1_conditional_u, APS3A_CDF_w2_conditional_u)[0]
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_pw1w2, initial_theta12, initial_theta13])

    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_z1 = matlab.double(z1.tolist())
    matlab_z2 = matlab.double(z2.tolist())
    matlab_us = matlab.double(us.tolist())
    
    try: #Estimate the copula parameters
        tmp_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_Vine_APS3A(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
    except matlab.engine.MatlabExecutionError: #If we have an error try other random starting values for theta12, theta13
        estimation_exit_bool = False
        i = 0
        while estimation_exit_bool == False:
            if i > random_param_trial_runs:
                estimation_exit_bool = True
            initial_theta12 = random.uniform(-0.49, 0.49)
            initial_theta13 = random.uniform(-0.49, 0.49)
            APS3A_CDF_w1_conditional_u = CDF_w1 + initial_theta12*CDF_simulated_us*CDF_w1*(4*CDF_w1**2 - 6*CDF_w1 + 2) + initial_theta12*CDF_w1*(CDF_simulated_us - 1)*(4*CDF_w1**2 - 6*CDF_w1 + 2)
            APS3A_CDF_w2_conditional_u = CDF_w2 + initial_theta13*CDF_simulated_us*CDF_w2*(4*CDF_w2**2 - 6*CDF_w2 + 2) + initial_theta13*CDF_w2*(CDF_simulated_us - 1)*(4*CDF_w2**2 - 6*CDF_w2 + 2)
            initial_pw1w2 = stats.kendalltau(APS3A_CDF_w1_conditional_u, APS3A_CDF_w2_conditional_u)[0]
            theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_pw1w2, initial_theta12, initial_theta13])
            matlab_theta0 = matlab.double(theta0.tolist())
            matlab_theta0.reshape((len(theta0),1))
            try:
                tmp_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_Vine_APS3A(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
                estimation_exit_bool = True
            except matlab.engine.MatlabExecutionError:
                i += 1
                continue
    
    tmp_theta = np.array(tmp_theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    theta = tmp_theta[:-3]
    u_hat = np.array(u_hat)
    non_par_u_hat = np.array(non_par_u_hat)
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2)   
    full_theta = np.concatenate([theta, np.array([tmp_theta[-2], tmp_theta[-1], tmp_theta[-3]])])
    return full_theta, theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2

def AppEstimate_Vine_APS3B(y, x1, x2, z1, z2, w1, w2, us, random_seed, eng, true_theta12, true_theta13):
    #Define initial values for maximization of the log-likelihood
    initial_theta12_dict = {0.3:0.1, 0.45:0.3, 0.8:0.4, 0.9:0.4}
    initial_theta13_dict = {0.1:0.05, 0.45:0.3, 0.7:0.4, -0.7:-0.4}
    initial_alpha = 0.5
    initial_beta1 = 0.5
    initial_beta2 = 0.5
    initial_sigma2u = 1
    initial_sigma2v = 1
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigma2w1 = 1
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_sigma2w2 = 1
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_gamma = 1
    initial_theta12 = initial_theta12_dict[true_theta12]
    initial_theta13 = initial_theta13_dict[true_theta13]
#    b = 1
#    a = -1
#    initial_arctanh_theta12 = np.arctanh((initial_theta12 - (b + a)/2)/((b - a)/2))
#    initial_arctanh_theta13 = np.arctanh((initial_theta13 - (b + a)/2)/((b - a)/2))
    
    #Conditional CDF transforms for allocative inefficiency correlation starting value
    CDF_simulated_us = 2*(stats.norm.cdf(np.sqrt(initial_sigma2u)*us[0,:], 0, np.sqrt(initial_sigma2u)) -0.5).ravel() #use the first random sample of u in the intial guess for pw1w2
    CDF_w1 = stats.norm.cdf(w1, loc = 0 , scale = np.sqrt(initial_sigma2w1)).ravel()
    CDF_w2 = stats.norm.cdf(w2, loc = 0 , scale = np.sqrt(initial_sigma2w2)).ravel()
#    APS3B_CDF_w1_conditional_u = np.where(CDF_w1 <= 0.5, CDF_w1 + (initial_theta12*CDF_w1)*(1 - CDF_w1)*(4*CDF_simulated_us - 1), CDF_w1 + (initial_theta12*CDF_w1)*(1 - CDF_w1)*(3 - 4*CDF_simulated_us))
#    APS3B_CDF_w2_conditional_u = np.where(CDF_w2 <= 0.5, CDF_w2 + (initial_theta13*CDF_w2)*(1 - CDF_w2)*(4*CDF_simulated_us - 1), CDF_w2 + (initial_theta13*CDF_w2)*(1 - CDF_w2)*(3 - 4*CDF_simulated_us))
    APS3B_CDF_w1_conditional_u = np.where(CDF_w1 <= 0.5, CDF_w1 + initial_theta12*CDF_simulated_us*(-2*CDF_w1**2 + CDF_w1) + initial_theta12*(-2*CDF_w1**2 + CDF_w1)*(CDF_simulated_us - 1), 
                                          CDF_w1 + initial_theta12*(CDF_simulated_us - 1)*(2*CDF_w1**2 - 3*CDF_w1 + 1) + initial_theta12*CDF_simulated_us*(2*CDF_w1**2 - 3*CDF_w1 + 1))
    APS3B_CDF_w2_conditional_u = np.where(CDF_w2 <= 0.5, CDF_w2 + initial_theta13*CDF_simulated_us*(-2*CDF_w2**2 + CDF_w2) + initial_theta13*(-2*CDF_w2**2 + CDF_w2)*(CDF_simulated_us - 1), 
                                          CDF_w2 + initial_theta13*(CDF_simulated_us - 1)*(2*CDF_w2**2 - 3*CDF_w2 + 1) + initial_theta13*CDF_simulated_us*(2*CDF_w2**2 - 3*CDF_w2 + 1))
    initial_pw1w2 = stats.kendalltau(APS3B_CDF_w1_conditional_u, APS3B_CDF_w2_conditional_u)[0]
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_pw1w2, initial_theta12, initial_theta13])

    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_z1 = matlab.double(z1.tolist())
    matlab_z2 = matlab.double(z2.tolist())
    matlab_us = matlab.double(us.tolist())

    try: #Estimate the copula parameters
        tmp_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_Vine_APS3B(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
    except matlab.engine.MatlabExecutionError: #If we have an error try other random starting values for theta12, theta13
        estimation_exit_bool = False
        i = 0
        while estimation_exit_bool == False:
            if i > random_param_trial_runs:
                estimation_exit_bool = True
            initial_theta12 = random.uniform(-0.99, 0.99)
            initial_theta13 = random.uniform(-0.99, 0.99)
            APS3B_CDF_w1_conditional_u = np.where(CDF_w1 <= 0.5, CDF_w1 + initial_theta12*CDF_simulated_us*(-2*CDF_w1**2 + CDF_w1) + initial_theta12*(-2*CDF_w1**2 + CDF_w1)*(CDF_simulated_us - 1), 
                                                  CDF_w1 + initial_theta12*(CDF_simulated_us - 1)*(2*CDF_w1**2 - 3*CDF_w1 + 1) + initial_theta12*CDF_simulated_us*(2*CDF_w1**2 - 3*CDF_w1 + 1))
            APS3B_CDF_w2_conditional_u = np.where(CDF_w2 <= 0.5, CDF_w2 + initial_theta13*CDF_simulated_us*(-2*CDF_w2**2 + CDF_w2) + initial_theta13*(-2*CDF_w2**2 + CDF_w2)*(CDF_simulated_us - 1), 
                                                  CDF_w2 + initial_theta13*(CDF_simulated_us - 1)*(2*CDF_w2**2 - 3*CDF_w2 + 1) + initial_theta13*CDF_simulated_us*(2*CDF_w2**2 - 3*CDF_w2 + 1))
            initial_pw1w2 = stats.kendalltau(APS3B_CDF_w1_conditional_u, APS3B_CDF_w2_conditional_u)[0]
            theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_lsigma2v, initial_lsigma2u, initial_lsigma2w1, initial_lsigma2w2, initial_gamma, initial_pw1w2, initial_theta12, initial_theta13])
            matlab_theta0 = matlab.double(theta0.tolist())
            matlab_theta0.reshape((len(theta0),1))
            try:
                tmp_theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2 = eng.nested_AppLoglikelihood_Vine_APS3B(matlab_y, matlab_x1, matlab_x2, matlab_z1, matlab_z2, matlab_us, matlab_theta0, random_seed, nargout = 5)
                estimation_exit_bool = True
            except matlab.engine.MatlabExecutionError:
                i += 1
                continue

    tmp_theta = np.array(tmp_theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    theta = tmp_theta[:-3]
    u_hat = np.array(u_hat)
    non_par_u_hat = np.array(non_par_u_hat)
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2)   
    full_theta = np.concatenate([theta, np.array([tmp_theta[-2], tmp_theta[-1], tmp_theta[-3]])])
    return full_theta, theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2

def model_estimation(y, x1, x2, z1, z2, w1, w2, us, true_theta, u, s, true_theta12, true_theta13):
    ### Eastimate SFA model params ###
    theta = {}
    theta_errors = {}
    theta_standard_errors = {}
    u_hat_errors = {}
    non_par_u_hat_errors = {}
    u_hat_conditional_w1w2_errors = {}

    #Independence between technical and allocative inefficiency 
    QMLE_full_coefs, QMLE_coefs, QMLE_sterr, QMLE_u_hat, QMLE_u_hat_conditional_w1w2 = AppEstimate_QMLE(y, x1, x2, z1, z2, eng)
    QMLE_error = QMLE_coefs - true_theta
    theta['QMLE'] = QMLE_full_coefs
    theta_errors['QMLE'] = QMLE_error
    theta_standard_errors['QMLE'] = QMLE_sterr
    u_hat_errors['QMLE'] = np.sqrt(np.mean(np.square(QMLE_u_hat - u)))
    non_par_u_hat_errors['QMLE'] = 0
    u_hat_conditional_w1w2_errors['QMLE'] = 0

#   #Gaussian copula
    gaussian_copula_full_coefs, gaussian_copula_coefs, gaussian_sterr, gaussian_copula_u_hat, gaussian_copula_non_par_u_hat, gaussian_copula_u_hat_conditional_w1w2 = AppEstimate_Gaussian_FMLE(y, x1, x2, z1, z2, w1, w2, us, s, eng)
    gaussian_copula_error = gaussian_copula_coefs - true_theta
    theta['Gaussian Copula'] = gaussian_copula_full_coefs
    theta_errors['Gaussian Copula'] = gaussian_copula_error
    theta_standard_errors['Gaussian Copula'] = gaussian_sterr
    u_hat_errors['Gaussian Copula'] = np.sqrt(np.mean(np.square(gaussian_copula_u_hat - u)))
    non_par_u_hat_errors['Gaussian Copula'] = np.sqrt(np.mean(np.square(gaussian_copula_non_par_u_hat - u)))
    u_hat_conditional_w1w2_errors['Gaussian Copula'] = np.sqrt(np.mean(np.square(gaussian_copula_u_hat_conditional_w1w2 - u)))
   
    #APS3B Copula 
    APS3B_copula_full_coefs, APS3B_copula_coefs, APS3B_sterr, APS3B_copula_u_hat, APS3B_copula_non_par_u_hat, APS3B_copula_u_hat_conditional_w1w2 = AppEstimate_APS3B(y, x1, x2, z1, z2, w1, w2, us, s, eng, true_theta12, true_theta13)
    APS3B_copula_error = APS3B_copula_coefs - true_theta
    theta['APS3B'] = APS3B_copula_full_coefs
    theta_errors['APS3B']= APS3B_copula_error
    theta_standard_errors['APS3B'] = APS3B_sterr
    u_hat_errors['APS3B'] = np.sqrt(np.mean(np.square(APS3B_copula_u_hat - u)))
    non_par_u_hat_errors['APS3B'] = np.sqrt(np.mean(np.square(APS3B_copula_non_par_u_hat - u)))
    u_hat_conditional_w1w2_errors['APS3B'] = np.sqrt(np.mean(np.square(APS3B_copula_u_hat_conditional_w1w2 - u)))  

    #APS3A Copula 
    APS3A_copula_full_coefs, APS3A_copula_coefs, APS3A_sterr, APS3A_copula_u_hat, APS3A_copula_non_par_u_hat, APS3A_copula_u_hat_conditional_w1w2 = AppEstimate_APS3A(y, x1, x2, z1, z2, w1, w2, us, s, eng, true_theta12, true_theta13)
    APS3A_copula_error = APS3A_copula_coefs - true_theta
    theta['APS3A'] = APS3A_copula_full_coefs
    theta_errors['APS3A'] = APS3A_copula_error
    theta_standard_errors['APS3A'] = APS3A_sterr
    u_hat_errors['APS3A'] = np.sqrt(np.mean(np.square(APS3A_copula_u_hat - u)))
    non_par_u_hat_errors['APS3A'] = np.sqrt(np.mean(np.square(APS3A_copula_non_par_u_hat - u)))
    u_hat_conditional_w1w2_errors['APS3A'] = np.sqrt(np.mean(np.square(APS3A_copula_u_hat_conditional_w1w2 - u)))

    #Vine APS3B Copula
    Vine_APS3B_copula_full_coefs, Vine_APS3B_copula_coefs, Vine_APS3B_sterr, Vine_APS3B_copula_u_hat, Vine_APS3B_copula_non_par_u_hat, Vine_APS3B_copula_u_hat_conditional_w1w2 = AppEstimate_Vine_APS3B(y, x1, x2, z1, z2, w1, w2, us, s, eng, true_theta12, true_theta13)
    Vine_APS3B_copula_error = Vine_APS3B_copula_coefs - true_theta
    theta['Vine_APS3B'] = Vine_APS3B_copula_full_coefs
    theta_errors['Vine_APS3B'] = Vine_APS3B_copula_error
    theta_standard_errors['Vine_APS3B'] = Vine_APS3B_sterr
    u_hat_errors['Vine_APS3B'] = np.sqrt(np.mean(np.square(Vine_APS3B_copula_u_hat - u)))
    non_par_u_hat_errors['Vine_APS3B'] = np.sqrt(np.mean(np.square(Vine_APS3B_copula_non_par_u_hat - u)))
    u_hat_conditional_w1w2_errors['Vine_APS3B'] = np.sqrt(np.mean(np.square(Vine_APS3B_copula_u_hat_conditional_w1w2 - u)))  
    
#    #Vine APS3A Copula
    Vine_APS3A_copula_full_coefs, Vine_APS3A_copula_coefs, Vine_APS3A_sterr, Vine_APS3A_copula_u_hat, Vine_APS3A_copula_non_par_u_hat, Vine_APS3A_copula_u_hat_conditional_w1w2 = AppEstimate_Vine_APS3A(y, x1, x2, z1, z2, w1, w2, us, s, eng, true_theta12, true_theta13)
    Vine_APS3A_copula_error = Vine_APS3A_copula_coefs - true_theta
    theta['Vine_APS3A'] = Vine_APS3A_copula_full_coefs
    theta_errors['Vine_APS3A'] = Vine_APS3A_copula_error
    theta_standard_errors['Vine_APS3A'] = Vine_APS3A_sterr
    u_hat_errors['Vine_APS3A'] = np.sqrt(np.mean(np.square(Vine_APS3A_copula_u_hat - u)))
    non_par_u_hat_errors['Vine_APS3A'] = np.sqrt(np.mean(np.square(Vine_APS3A_copula_non_par_u_hat - u)))
    u_hat_conditional_w1w2_errors['Vine_APS3A'] = np.sqrt(np.mean(np.square(Vine_APS3A_copula_u_hat_conditional_w1w2 - u))) 
    
    print('Completed Simulation #{}'.format(s))
    return theta, theta_errors, theta_standard_errors, u_hat_errors, non_par_u_hat_errors, u_hat_conditional_w1w2_errors

def simulate_Vine_APS2B(theta12, theta13, pw1w2, n, random_seed):
    sample = eng.simulate_sample_Vine_APS2B(theta12, theta13, pw1w2, n, random_seed, nargout = 1)
    sample = np.array(sample)
    return sample

def global_variable_initializer():
    global base_file_path, eng, random_param_trial_runs
    random_param_trial_runs = 1000
    base_file_path = r'C:\Users\rjam3686\Dropbox (Sydney Uni)\Vine Copula SFA Project'
    eng = matlab.engine.start_matlab()
    eng.addpath(base_file_path, nargout=0)
    
# ============================================================================= 
def main():
    global_variable_initializer()
    #Setup simulation parameters
    estimation_methods = ['QMLE', 'Gaussian Copula', 'APS3A', 'APS3B', 'Vine_APS3A', 'Vine_APS3B']
    num_iterations = 1000
    num_integral_sample = 100
    true_sigma2u = 1 #variance of technical inefficiency term
    true_sigma2v = 1 #variance of symmetric random noise term 
    true_sigma2w1 = 1 #variance of the inefficiency term for production unit 1
    true_sigma2w2 = 1 #variance of the inefficiency term for production unit 2
    true_gamma = 1 #coefficient for the endogenous regressor equations: x_{j} = \gamma z_{j} + w_{j}, j = 1,2
    true_alpha = 0.5
    true_beta1 = 0.5    
    true_beta2 = 0.5
    
    #Setup process pool and initialize with global variables
    num_processes = multiprocessing.cpu_count() - 1
    processing_pool = multiprocessing.Pool(num_processes, global_variable_initializer)

    if os.path.exists(r'{}\sample_correlation_information.txt'.format(base_file_path)):
        os.remove(r'{}\sample_correlation_information.txt'.format(base_file_path))

    #Simulations
    for n in [500, 1000]:
        theta_results2 = []
        RMSE_results2 = []
        sterr_results2 = []
        #for Vine APS2A abs(theta12) and abs(theta13) must be <= 0.5
        #for Vine APS2B abs(theta12) and abs(theta13) must be <= 1
        for correlation_params in tqdm([(0.3, 0.1, 0.4), (0.45, 0.45, 0.4), (0.8, 0.7, 0.4)]):
            true_theta12, true_theta13, true_pw1w2 = correlation_params
            estimation_method_theta = {model:pd.DataFrame(columns = ['alpha', 'beta1', 'beta2', 'sigma2u', 'sigma2v', 'sigma2w1', 'sigma2w2', 'gamma', 'theta12', 'theta13', 'pw1w2'], index = range(num_iterations)) for model in estimation_methods}
            estimation_method_error = {model:pd.DataFrame(columns = ['alpha', 'beta1', 'beta2', 'sigma2u', 'sigma2v', 'sigma2w1', 'sigma2w2', 'gamma'], index = range(num_iterations)) for model in estimation_methods}
            estimation_method_standard_errors = {model:pd.DataFrame(columns = ['alpha', 'beta1', 'beta2', 'sigma2u', 'sigma2v', 'sigma2w1', 'sigma2w2', 'gamma'], index = range(num_iterations)) for model in estimation_methods}
            estimation_method_u_hat_errors = {model:pd.DataFrame(columns = ['u_hat'], index = range(num_iterations)) for model in estimation_methods}
            estimation_method_non_par_u_hat_errors = {model:pd.DataFrame(columns = ['non_par_u_hat'], index = range(num_iterations)) for model in estimation_methods}
            estimation_method_u_hat_conditional_w1w2_errors = {model:pd.DataFrame(columns = ['u_hat_conditional_w1w2'], index = range(num_iterations)) for model in estimation_methods}
            true_theta = np.array([true_alpha, true_beta1, true_beta2, true_sigma2v, true_sigma2u, true_sigma2w1, true_sigma2w2, true_gamma])
            mapping_list = []
            
            corr_u_absw1 = []; corr_u_absw2 = [] 
            corr_u_w1 = []; corr_u_w2 = [] 
            corr_w1w2 = []
            for s in tqdm(range(num_iterations)):
                s = s + 1
                #Simulate sample from APS3B copula
                sample = simulate_Vine_APS2B(theta12 = true_theta12, theta13 = true_theta13, pw1w2 = true_pw1w2, n = n, random_seed = s)
                #Set up the simulated data 
                u = stats.norm.ppf((sample[:,0]+1)/2, 0, 1).reshape(-1,1) #simulated technical inefficiency terms, based upon the half normal distribution (abs of normal RV's)
                w1 = stats.norm.ppf(sample[:,1], 0, 1).reshape(-1,1) #simulated asymmetric alocative inefficiency terms
                w2 = stats.norm.ppf(sample[:,2], 0, 1).reshape(-1,1) #simulated asymmetric alocative inefficiency terms
                v = stats.norm.rvs(loc = 0, scale = 1, size = (n,1), random_state = s+4)
                z1 = stats.chi2.rvs(df = 2, size = (n,1), random_state = s+2); #Endogenous regressor
                z2 = stats.chi2.rvs(df = 2, size = (n,1), random_state = s+3); #Endogenous regressor
                x1 = z1 * true_gamma + w1 #generate input based upon endogenous variables
                x2 = z2 * true_gamma + w2 #generate input based upon endogenous variables
                y = true_alpha + true_beta1 * x1 + true_beta2 * x2 + v - u #simulated output
                    
                #Some stats about the DGP
                corr_u_absw1.append(np.corrcoef(np.column_stack([u, np.abs(w1)]).T)[0,1])
                corr_u_absw2.append(np.corrcoef(np.column_stack([u, np.abs(w2)]).T)[0,1])
                corr_u_w1.append(np.corrcoef(np.column_stack([u, w1]).T)[0,1])
                corr_u_w2.append(np.corrcoef(np.column_stack([u, w2]).T)[0,1])
                corr_w1w2.append(np.corrcoef(np.column_stack([w1, w2]).T)[0,1])
                
                us = stats.norm.ppf((np.random.uniform(size = (num_integral_sample, n))+1)/2, loc = 0, scale = 1) #standard Normal random variables for simulation of the integral.
#                us_ = stats.norm.ppf((np.random.uniform(size = (1, n))+1)/2, loc = 0, scale = 1) #standard Normal random variables for simulation of the integral.
#                us = np.array([us_.ravel(),]*num_integral_sample)
                mapping_list.append([y, x1, x2, z1, z2, w1, w2, us, true_theta, u, s, true_theta12, true_theta13])
 
            #export information about the sample correlations generated in the simulation 
            with open (r'{}\sample_correlation_information.txt'.format(base_file_path), 'a+') as sample_correlation_info:
                print('n = {}, theta12 = {}, theta13 = {}, pw1w2 = {}'.format(n, true_theta12, true_theta13, true_pw1w2), file = sample_correlation_info)
                print('\t corr(u, w1) = {}'.format(round(np.mean(corr_u_w1), 4)), file = sample_correlation_info)
                print('\t corr(u, w2) = {}'.format(round(np.mean(corr_u_w2), 4)), file = sample_correlation_info)
                print('\t corr(u, abs(w1)) = {}'.format(round(np.mean(corr_u_absw1), 4)), file = sample_correlation_info)
                print('\t corr(u, abs(w2)) = {}'.format(round(np.mean(corr_u_absw2), 4)), file = sample_correlation_info)
                print('\t corr(w1, w2) = {}'.format(round(np.mean(corr_w1w2), 4)), file = sample_correlation_info)
                print('\t Std corr(u, abs(w1)) = {}'.format(round(np.std(corr_u_absw1), 4)), file = sample_correlation_info)
                print('\t std corr(u, abs(w2)) = {}'.format(round(np.std(corr_u_absw2), 4)), file = sample_correlation_info)

            #Estimate models
            results = processing_pool.starmap(model_estimation, mapping_list)
#            results= []
#            for i in tqdm(range(len(mapping_list))):
#                 results.append(model_estimation(y = mapping_list[i][0], x1 = mapping_list[i][1], x2 = mapping_list[i][2], z1 = mapping_list[i][3], z2 = mapping_list[i][4], w1 = mapping_list[i][5], w2 = mapping_list[i][6], us = mapping_list[i][7], true_theta = mapping_list[i][8], u = mapping_list[i][9], s = mapping_list[i][10], true_theta12 = mapping_list[i][11], true_theta13 = mapping_list[i][12]))            
            
            for i in range(len(results)):
                for model in estimation_methods:
                    estimation_method_theta[model].iloc[i,:] = results[i][0][model]
                    estimation_method_error[model].iloc[i,:] = results[i][1][model]
                    estimation_method_standard_errors[model].iloc[i,:] = results[i][2][model]
                    estimation_method_u_hat_errors[model].iloc[i] = results[i][3][model]
                    estimation_method_non_par_u_hat_errors[model].iloc[i] = results[i][4][model]
                    estimation_method_u_hat_conditional_w1w2_errors[model].iloc[i] = results[i][5][model]
    
            estimation_method_avg_theta = {}
            estimation_method_RMSE = {}
            estimation_method_sterr = {}
            for model in estimation_methods:
                theta_frame = np.sqrt(np.square(estimation_method_theta[model]).mean(axis = 0))
                theta_frame['Simulated $\theta$'] = '($\theta_12 = {}$, $\theta_13 = {}$, $\rho_w_1w_2 = {}$)'.format(true_theta12, true_theta13, true_pw1w2)
                theta_frame = theta_frame.reindex(['Simulated $\theta$'] + theta_frame.index[:-1].tolist())
                
                RMSE_frame = np.sqrt(np.square(estimation_method_error[model]).mean(axis = 0))
                RMSE_frame['Simulated $\theta$'] = '($\theta_12 = {}$, $\theta_13 = {}$, $\rho_w_1w_2 = {}$)'.format(true_theta12, true_theta13, true_pw1w2)
                RMSE_frame = RMSE_frame.reindex(['Simulated $\theta$'] + RMSE_frame.index[:-1].tolist())
                RMSE_frame['Total'] = np.sum(np.sqrt(np.square(estimation_method_error[model]).mean(axis = 0)))
                
                sterr_frame = estimation_method_standard_errors[model].mean(axis = 0)
                sterr_frame['Simulated $\theta$'] = '($\theta_12 = {}$, $\theta_13 = {}$, $\rho_w_1w_2 = {}$)'.format(true_theta12, true_theta13, true_pw1w2)
                sterr_frame = sterr_frame.reindex(['Simulated $\theta$'] + sterr_frame.index[:-1].tolist())
                
                RMSE_frame.loc['u_hat'] = np.sqrt(np.square(estimation_method_u_hat_errors[model]).mean(axis = 0))['u_hat']
                RMSE_frame.loc['non_par_u_hat'] = np.sqrt(np.square(estimation_method_non_par_u_hat_errors[model]).mean(axis = 0))['non_par_u_hat']
                RMSE_frame.loc['u_hat_conditional_w1w2'] = np.sqrt(np.square(estimation_method_u_hat_conditional_w1w2_errors[model]).mean(axis = 0))['u_hat_conditional_w1w2']
    
                estimation_method_RMSE[model] = RMSE_frame
                estimation_method_sterr[model] = sterr_frame 
                estimation_method_avg_theta[model] = theta_frame
                
            RMSE_results2.append(estimation_method_RMSE)
            sterr_results2.append(estimation_method_sterr)
            theta_results2.append(estimation_method_avg_theta)
            
            #Export intermediate results
            pd.concat(estimation_method_RMSE, axis = 1).to_csv(r'{}\parameter_RMSE_simulation_results_n={}_theta12={}_theta13={}pw1w2={}.csv'.format(base_file_path, n, true_theta12, true_theta13, true_pw1w2), index = True)
            pd.concat(estimation_method_sterr, axis = 1).to_csv(r'{}\parameter_sterr_simulation_results_n={}_theta12={}_theta13={}pw1w2={}.csv'.format(base_file_path, n, true_theta12, true_theta13, true_pw1w2), index = True)
            pd.concat(estimation_method_avg_theta, axis = 1).to_csv(r'{}\parameter_avg_theta_simulation_results_n={}_theta12={}_theta13={}pw1w2={}.csv'.format(base_file_path, n, true_theta12, true_theta13, true_pw1w2), index = True)
            
        #Concatenate each set of results for different sample sizes. 
        rho_concatenated_results_parameter_theta = {model:pd.concat([x[model] for x in theta_results2]) for model in estimation_methods}
        rho_concatenated_results_parameter_RMSE = {model:pd.concat([x[model] for x in RMSE_results2]) for model in estimation_methods}
        rho_concatenated_results_paramneter_sterr = {model:pd.concat([x[model] for x in sterr_results2]) for model in estimation_methods}
        final_results_parameter_theta_n = pd.concat(rho_concatenated_results_parameter_theta.values(), axis = 1)
        final_results_parameter_RMSE_n = pd.concat(rho_concatenated_results_parameter_RMSE.values(), axis = 1)
        final_results_parameter_sterr_n = pd.concat(rho_concatenated_results_paramneter_sterr.values(), axis = 1)
        final_results_parameter_theta_n.columns = list(rho_concatenated_results_parameter_theta.keys())
        final_results_parameter_RMSE_n.columns = list(rho_concatenated_results_parameter_RMSE.keys())
        final_results_parameter_sterr_n.columns = list(final_results_parameter_sterr_n.keys())
        final_results_parameter_theta_n.to_csv(r'{}\parameter_avg_theta_simulation_results_n={}.csv'.format(base_file_path, n), index = True)
        final_results_parameter_RMSE_n.to_csv(r'{}\parameter_RMSE_simulation_results_n={}.csv'.format(base_file_path, n), index = True)
        final_results_parameter_sterr_n.to_csv(r'{}\parameter_sterr_simulation_results_n={}.csv'.format(base_file_path, n), index = True)
    eng.quit()
if __name__ == '__main__':  
    main()