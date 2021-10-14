# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 08:35:27 2021

@author: Robert James
"""
import matlab.engine
import numpy as np 
import pandas as pd
import multiprocessing
from scipy import stats
from tqdm import tqdm 

def AppEstimate_ALS77(y, x1, x2, x3):
    #Starting values for MLE
    initial_alpha = -11
    initial_beta1 = 0.03
    initial_beta2 = 1.1
    initial_beta3 = -0.01
    initial_sigma2u = 0.01 #variance of technical inefficiency
    initial_sigma2v = 0.0003 #variance of random noise
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3, initial_lsigma2v, initial_lsigma2u])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    
    LL, theta, sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_ALS77(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_theta0, nargout = 8)
    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(u_hat))
    non_par_u_hat = np.mean(np.array(non_par_u_hat))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_conditional_w1w2 = np.array(u_hat_conditional_w1w2) 
    V_u_hat_conditional_w1w2 = 0
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))
    
    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    #Reorder parameter vector for table
    final_theta = np.concatenate([theta[:4], np.array([None, None]), theta[4:], np.array([None, None, None, None, None, None, None, None, None, None])]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:4], np.array([None, None]), sterr[4:], np.array([None, None, None, None, None, None, None, None, None, None])]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def AppEstimate_SL80(y, x1, x2, x3, p1, p2, p3):
    #Starting values for MLE
    initial_alpha = -11.68
    initial_beta1 = 0.129
    initial_beta2 = 0.974
    initial_beta3 = 0.063
    initial_sigma2u = 0.014 #variance of technical inefficiency
    initial_sigma2v = 0.002 #variance of random noise
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_sigmau2 = 0.036 #covariance of u and w2
    initial_sigmau3 = 0.005 #covariance of u and w3
    initial_sigma22 = 0.337 #variance of w2
    initial_sigma33 = 0.59 #variance of w2
    initial_sigma23 = 0.20 #covariance of w2 and w3
    initial_mu1 = 0.88 #mean of the normal distribution for w1
    initial_mu2 = 0.45 #mean of the normal distribution for w2
     
    initial_Sigma = np.array([[initial_sigma2u, initial_sigmau2, initial_sigmau3], 
                              [initial_sigmau2, initial_sigma22, initial_sigma23], 
                              [initial_sigmau3, initial_sigma23, initial_sigma33]]) #Initial covariance matrix eq. (A.2) in SL80
    
    #Use Cholesky decomposition of covariance matrix in optimzation since L has real positive diagonal entries (variances) so we dont need to constrain to positive numbers
    tril_initial_Sigma = np.tril(np.linalg.cholesky(initial_Sigma)).flatten('F')
    tril_initial_Sigma = tril_initial_Sigma[tril_initial_Sigma !=0] #columnwise flattened non zero elements of the Cholesky decomposition of Sigma
    theta0 = np.concatenate([np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3]), tril_initial_Sigma, np.array([initial_lsigma2v, initial_mu1, initial_mu2])])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    matlab_p1 = matlab.double(p1.tolist())
    matlab_p2 = matlab.double(p2.tolist())
    matlab_p3 = matlab.double(p3.tolist())
    LL, theta, sterr, u_hat, V_u_hat, non_par_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_SL80(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_p1, matlab_p2, matlab_p3, matlab_theta0, nargout = 9)
    
    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(np.array(u_hat)))
    non_par_u_hat = np.mean(np.array(np.array(non_par_u_hat)))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_conditional_w1w2 = np.mean(np.array(u_hat_conditional_w1w2))
    V_u_hat_conditional_w1w2 = np.mean(np.array(V_u_hat_conditional_w1w2))
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))
    
    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    #Derive pw1w2 from cov(w1,w2) and std(w1), std(w2)
    covw1w2 = theta[4:-3][4]
    covuw1 = theta[4:-3][0]
    covuw2 = theta[4:-3][1]
    std_covw1w2 = sterr[4:-3][4]
    std_covuw1 = sterr[4:-3][0]
    std_covuw2 = sterr[4:-3][1]
    varw1 = theta[4:-3][3]
    varw2 = theta[4:-3][5]
    pw1w2 = covw1w2/(np.sqrt(varw1)*np.sqrt(varw2))
    
    #Reorder parameter vector for table
    final_theta = np.concatenate([theta[:4], np.array([theta[-2], theta[-1], theta[4:-3][0], theta[-3], theta[4:-3][3], theta[4:-3][5], covw1w2, None, covuw1, covuw2]), np.array([None, None, None, None])]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:4], np.array([sterr[-2], sterr[-1], sterr[4:-3][0], sterr[-3], sterr[4:-3][3], sterr[4:-3][5], std_covw1w2, None, std_covuw1, std_covuw2]), np.array([None, None, None, None])]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def AppEstimate_APS16(y, x1, x2, x3, p1, p2, p3):
    #Starting values for MLE
    initial_alpha = -12
    initial_beta1 = 0.11
    initial_beta2 = 0.999
    initial_beta3 = 0.06
    initial_sigma2u = 0.01 #variance of technical inefficiency
    initial_sigma2v = 0.005 #variance of random noise
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_sigmau2 = 0.036 #covariance of v and w2
    initial_sigmau3 = 0.005 #covariance of v and w3
    initial_sigma22 = 0.337 #variance of w2
    initial_sigma33 = 0.59 #variance of w2
    initial_sigma23 = 0.20 #covariance of w2 and w3
     
    initial_Sigma = np.array([[initial_sigma2v, initial_sigmau2, initial_sigmau3], 
                              [initial_sigmau2, initial_sigma22, initial_sigma23], 
                              [initial_sigmau3, initial_sigma23, initial_sigma33]]) #Initial covariance matrix eq. (A.2) in SL80
    
    #Use Cholesky decomposition of covariance matrix in optimzation since L has real positive diagonal entries (variances) so we dont need to constrain to positive numbers
    tril_initial_Sigma = np.tril(np.linalg.cholesky(initial_Sigma)).flatten('F')
    tril_initial_Sigma = tril_initial_Sigma[tril_initial_Sigma !=0] #columnwise flattened non zero elements of the Cholesky decomposition of Sigma
    theta0 = np.concatenate([np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3]), tril_initial_Sigma, np.array([initial_lsigma2u])])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    matlab_p1 = matlab.double(p1.tolist())
    matlab_p2 = matlab.double(p2.tolist())
    matlab_p3 = matlab.double(p3.tolist())
    LL, theta, sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_APS16(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_p1, matlab_p2, matlab_p3, matlab_theta0, nargout = 9)
    
    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(u_hat))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_conditional_w1w2 = np.mean(np.array(u_hat_conditional_w1w2)) 
    V_u_hat_conditional_w1w2 = np.mean(np.array(V_u_hat_conditional_w1w2)) 
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))
    non_par_u_hat = np.mean(np.array(non_par_u_hat))
    
    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    #Derive pw1w2 from cov(w1,w2) and std(w1), std(w2)
    covw1w2 = theta[4:-1][4]
    covuw1 = theta[4:-1][0]
    covuw2 = theta[4:-1][1]
    std_covw1w2 = sterr[4:-1][4]
    std_covuw1 = sterr[4:-1][0]
    std_covuw2 = sterr[4:-1][1]
    stdw1 = np.sqrt(theta[4:-1][3])
    stdw2 = np.sqrt(theta[4:-1][5])
    pw1w2 = covw1w2/(stdw1*stdw2)
    #Reorder parameter vector for table
    final_theta = np.concatenate([theta[:4], np.array([None, None, theta[-1], theta[4:-1][0], theta[4:-1][3], theta[4:-1][5], covw1w2, None, None, None, covuw1, covuw2]), np.array([None, None])]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:4], np.array([None, None, sterr[-1], sterr[4:-1][0], sterr[4:-1][3], sterr[4:-1][5], std_covw1w2, None, None, None, std_covuw1, std_covuw2]), np.array([None, None])]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def AppEstimate_APS3A(y, x1, x2, x3, p1, p2, p3, us):
    #Starting values for MLE
    initial_alpha = -11.6
    initial_beta1 = 0.06
    initial_beta2 = 1.04
    initial_beta3 = 0.06
    initial_sigma2u = 0.014 #variance of technical inefficiency
    initial_sigma2v = 0.002 #variance of random noise
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    theta12 = 0 #APS c12 copula parameter
    theta13 = -0 #APS c13 copula parameter
    initial_sigma2w1 = 0.33 #variance of w1
    initial_sigma2w2 = 0.59 #variance of w2
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_mu2 = 1.3 #mean of normal distribution for w1
    initial_mu3 = 1 #mean of normal distribution for w2
    #Find starting value for rho in gaussian copula for w1w2
    B2 = p2 - p1 + np.log(initial_beta1) - np.log(initial_beta2)
    B3 = p3 - p1 + np.log(initial_beta1) - np.log(initial_beta3)
    w1 = x1 - x2 - B2
    w2 = x1 - x3 - B3
    pw1w2 = stats.kendalltau(stats.norm.cdf(w1, loc = initial_mu2, scale = np.sqrt(initial_sigma2w1)), stats.norm.cdf(w2, loc = initial_mu3, scale = np.sqrt(initial_sigma2w2)))[0]
    #Initial parameter vector
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3, initial_mu2, initial_mu3, 
                                       initial_lsigma2u, initial_lsigma2v, initial_lsigma2w1, initial_lsigma2w2, pw1w2, theta12, theta13])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    matlab_p1 = matlab.double(p1.tolist())
    matlab_p2 = matlab.double(p2.tolist())
    matlab_p3 = matlab.double(p3.tolist())
    matlab_us = matlab.double(us.tolist())
    LL, theta, sterr, u_hat, V_u_hat, u_hat_non_par, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_APS3A_unconstrained(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_p1, matlab_p2, matlab_p3, matlab_us, matlab_theta0, nargout = 9)

    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(u_hat))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_non_par = np.mean(np.array(u_hat_non_par))
    u_hat_conditional_w1w2 = np.mean(np.array(u_hat_conditional_w1w2))
    V_u_hat_conditional_w1w2 = np.mean(np.array(V_u_hat_conditional_w1w2))
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))
    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    final_theta = np.concatenate([theta[:-3], np.array([None, theta[-3], None, None, None, None]), theta[-2:]]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:-3], np.array([None, sterr[-3], None, None, None, None]), sterr[-2:]]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, u_hat_non_par, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def AppEstimate_APS3B(y, x1, x2, x3, p1, p2, p3, us):
    #Starting values for MLE
    initial_alpha = -11.6
    initial_beta1 = 0.06
    initial_beta2 = 1.04
    initial_beta3 = 0.06
    initial_sigma2u = 0.014 #variance of technical inefficiency
    initial_sigma2v = 0.002 #variance of random noise
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_theta12 = 0.2 #APS c12 copula parameter
    initial_theta13 = -0.2 #APS c13 copula parameter
    initial_sigma2w1 = 0.33 #variance of w1
    initial_sigma2w2 = 0.59 #variance of w2
    initial_lsigma2w1 = np.log(initial_sigma2w1)
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_mu2 = 1.55 #mean of normal distribution for w1
    initial_mu3 = 1.05 #mean of normal distribution for w2
    #Find starting value for rho in gaussian copula for w1w2
    B2 = p2 - p1 + np.log(initial_beta1) - np.log(initial_beta2)
    B3 = p3 - p1 + np.log(initial_beta1) - np.log(initial_beta3)
    w1 = x1 - x2 - B2
    w2 = x1 - x3 - B3
    initial_pw1w2 = stats.kendalltau(stats.norm.cdf(w1, loc = initial_mu2, scale = np.sqrt(initial_sigma2w1)), stats.norm.cdf(w2, loc = initial_mu3, scale = np.sqrt(initial_sigma2w2)))[0]
    #Initial parameter vector
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3, initial_mu2, initial_mu3, 
                                       initial_lsigma2u, initial_lsigma2v, initial_lsigma2w1, initial_lsigma2w2, initial_pw1w2, initial_theta12, initial_theta13])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    matlab_p1 = matlab.double(p1.tolist())
    matlab_p2 = matlab.double(p2.tolist())
    matlab_p3 = matlab.double(p3.tolist())
    matlab_us = matlab.double(us.tolist())
    LL, theta, sterr, u_hat, V_u_hat, u_hat_non_par, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_APS3B_unconstrained(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_p1, matlab_p2, matlab_p3, matlab_us, matlab_theta0, nargout = 9)

    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(u_hat))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_non_par = np.mean(np.array(u_hat_non_par))
    u_hat_conditional_w1w2 = np.mean(np.array(u_hat_conditional_w1w2))
    V_u_hat_conditional_w1w2 = np.mean(np.array(V_u_hat_conditional_w1w2))  
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))

    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    final_theta = np.concatenate([theta[:-3], np.array([None, theta[-3], None, None, None, None]), theta[-2:]]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:-3], np.array([None, sterr[-3], None, None, None, None]), sterr[-2:]]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, u_hat_non_par, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def AppEstimate_Vine_APS3A(y, x1, x2, x3, p1, p2, p3, us):
    #Starting values for MLE
    initial_alpha = -11.6
    initial_beta1 = 0.06
    initial_beta2 = 1.04
    initial_beta3 = 0.06
    initial_sigma2u = 0.014 #variance of technical inefficiency
    initial_sigma2v = 0.002 #variance of random noise
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_theta12 = 0.2 #APS c12 copula parameter
    initial_theta13 = -0.2 #APS c13 copula parameter
    initial_sigma2w2 = 0.33 #variance of w2
    initial_sigma2w3 = 0.59 #variance of w3
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_lsigma2w3 = np.log(initial_sigma2w3)
    initial_mu2 = 2 #mean of normal distribution for w1
    initial_mu3 = 0.5 #mean of normal distribution for w2
    #Find starting value for rho in gaussian copula for w1w2
    B2 = p2 - p1 + np.log(initial_beta1) - np.log(initial_beta2)
    B3 = p3 - p1 + np.log(initial_beta1) - np.log(initial_beta3)
    w2 = x1 - x2 - B2
    w3 = x1 - x3 - B3
    
    #Conditional CDF transforms for allocative inefficiency correlation starting value
    CDF_simulated_us = 2*(stats.norm.cdf(np.sqrt(initial_sigma2u)*us[0,:], 0, np.sqrt(initial_sigma2u)) -0.5).ravel() #use the first random sample of u in the intial guess for pw1w2
    CDF_w2 = stats.norm.cdf(w2, loc = initial_mu2, scale = np.sqrt(initial_sigma2w2)).ravel()
    CDF_w3 = stats.norm.cdf(w3, loc = initial_mu3 , scale = np.sqrt(initial_sigma2w3)).ravel()
    APS3A_CDF_w2_conditional_u = CDF_w2 + (initial_theta12*CDF_w2)*(1 - CDF_w2)*(2*(6*CDF_simulated_us - 6*CDF_simulated_us**2 - 1))
    APS3A_CDF_w3_conditional_u = CDF_w3 + (initial_theta13*CDF_w3)*(1 - CDF_w3)*(2*(6*CDF_simulated_us - 6*CDF_simulated_us**2 - 1))
    initial_pw1w2 = stats.kendalltau(APS3A_CDF_w2_conditional_u, APS3A_CDF_w3_conditional_u)[0]
#    initial_pw1w2 = stats.kendalltau(w2, w3)[0]

    #Variable transformations for bounded theta12 and theta13
#    b = 0.5
#    a = -0.5
#    initial_arctanh_theta12 = np.arctanh((initial_theta12 - (b + a)/2)/((b - a)/2))
#    initial_arctanh_theta13 = np.arctanh((initial_theta13 - (b + a)/2)/((b - a)/2))
#    initial_tan_theta12 = np.tan((initial_theta12 - (b+a)/2)/(((b-a)/2)*2/np.pi))
#    initial_tan_theta13 = np.tan((initial_theta13 - (b+a)/2)/(((b-a)/2)*2/np.pi))

    #Initial parameter vector
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3, initial_mu2, initial_mu3, 
                                       initial_lsigma2u, initial_lsigma2v, initial_lsigma2w2, initial_lsigma2w3, initial_pw1w2, initial_theta12, initial_theta13])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    matlab_p1 = matlab.double(p1.tolist())
    matlab_p2 = matlab.double(p2.tolist())
    matlab_p3 = matlab.double(p3.tolist())
    matlab_us = matlab.double(us.tolist())
    LL, theta, sterr, u_hat, V_u_hat, u_hat_non_par, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_Vine_APS3A_unconstrained(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_p1, matlab_p2, matlab_p3, matlab_us, matlab_theta0, nargout = 9)

    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(u_hat))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_non_par = np.mean(np.array(u_hat_non_par))
    u_hat_conditional_w1w2 = np.mean(np.array(u_hat_conditional_w1w2))
    V_u_hat_conditional_w1w2 = np.mean(np.array(V_u_hat_conditional_w1w2))
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))
    
    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    final_theta = np.concatenate([theta[:-3], np.array([None, theta[-3], None, None, None, None]), theta[-2:]]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:-3], np.array([None, sterr[-3], None, None, None, None]), sterr[-2:]]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, u_hat_non_par, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def AppEstimate_Vine_APS3B(y, x1, x2, x3, p1, p2, p3, us):
    #Starting values for MLE
    initial_alpha = -11.6
    initial_beta1 = 0.045
    initial_beta2 = 1.04
    initial_beta3 = 0.06
    initial_sigma2u = 0.014 #variance of technical inefficiency
    initial_sigma2v = 0.002 #variance of random noise
    initial_lsigma2u = np.log(initial_sigma2u)
    initial_lsigma2v = np.log(initial_sigma2v)
    initial_theta12 = 0.2 #APS c12 copula parameter
    initial_theta13 = -0.2 #APS c13 copula parameter
    initial_sigma2w2 = 0.33 #variance of w1
    initial_sigma2w3 = 0.59 #variance of w2
    initial_lsigma2w2 = np.log(initial_sigma2w2)
    initial_lsigma2w3 = np.log(initial_sigma2w3)
    initial_mu2 = 2 #mean of normal distribution for w1
    initial_mu3 = 0.5 #mean of normal distribution for w2
    #Find starting value for rho in gaussian copula for w1w2
    B2 = p2 - p1 + np.log(initial_beta1) - np.log(initial_beta2)
    B3 = p3 - p1 + np.log(initial_beta1) - np.log(initial_beta3)
    w2 = x1 - x2 - B2
    w3 = x1 - x3 - B3
    
    #Conditional CDF transforms for allocative inefficiency correlation starting value
    CDF_simulated_us = 2*(stats.norm.cdf(np.sqrt(initial_sigma2u)*us[0,:], 0, np.sqrt(initial_sigma2u)) -0.5).ravel() #use the first random sample of u in the intial guess for pw1w2
    CDF_w2 = stats.norm.cdf(w2, loc = initial_mu2 , scale = np.sqrt(initial_sigma2w2)).ravel()
    CDF_w3 = stats.norm.cdf(w3, loc = initial_mu3 , scale = np.sqrt(initial_sigma2w3)).ravel()
    APS3B_CDF_w2_conditional_u = np.where(CDF_w2 <= 0.5, CDF_w2 + (initial_theta12*CDF_w2)*(1 - CDF_w2)*(4*CDF_simulated_us - 1), CDF_w2 + (initial_theta12*CDF_w2)*(1 - CDF_w2)*(3 - 4*CDF_simulated_us))
    APS3B_CDF_w3_conditional_u = np.where(CDF_w3 <= 0.5, CDF_w3 + (initial_theta13*CDF_w3)*(1 - CDF_w3)*(4*CDF_simulated_us - 1), CDF_w3 + (initial_theta12*CDF_w3)*(1 - CDF_w3)*(3 - 4*CDF_simulated_us))
    initial_pw1w2 = stats.kendalltau(APS3B_CDF_w2_conditional_u, APS3B_CDF_w3_conditional_u)[0]    

    #Variable transformations for bounded theta12 and theta13
#    b = 1
#    a = -1
#    initial_arctanh_theta12 = np.arctanh((initial_theta12 - (b + a)/2)/((b - a)/2))
#    initial_arctanh_theta13 = np.arctanh((initial_theta13 - (b + a)/2)/((b - a)/2))
#    initial_tan_theta12 = np.tan((initial_theta12 - (b+a)/2)/(((b-a)/2)*2/np.pi))
#    initial_tan_theta13 = np.tan((initial_theta13 - (b+a)/2)/(((b-a)/2)*2/np.pi))

    #Initial parameter vector
    theta0 = np.array([initial_alpha, initial_beta1, initial_beta2, initial_beta3, initial_mu2, initial_mu3, 
                                       initial_lsigma2u, initial_lsigma2v, initial_lsigma2w2, initial_lsigma2w3, initial_pw1w2, initial_theta12, initial_theta13])
    
    matlab_theta0 = matlab.double(theta0.tolist())
    matlab_theta0.reshape((len(theta0),1))
    matlab_y = matlab.double(y.tolist())
    matlab_x1 = matlab.double(x1.tolist())
    matlab_x2 = matlab.double(x2.tolist())
    matlab_x3 = matlab.double(x3.tolist())
    matlab_p1 = matlab.double(p1.tolist())
    matlab_p2 = matlab.double(p2.tolist())
    matlab_p3 = matlab.double(p3.tolist())
    matlab_us = matlab.double(us.tolist())
    LL, theta, sterr, u_hat, V_u_hat, u_hat_non_par, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat = eng.nested_AppLoglikelihood_Vine_APS3B_unconstrained(matlab_y, matlab_x1, matlab_x2, matlab_x3, matlab_p1, matlab_p2, matlab_p3, matlab_us, matlab_theta0, nargout = 9)

    theta = np.array(theta).ravel().astype(float)
    sterr = np.array(sterr).ravel().astype(float)
    u_hat = np.mean(np.array(u_hat))
    V_u_hat = np.mean(np.array(V_u_hat))
    u_hat_non_par = np.mean(np.array(u_hat_non_par))
    u_hat_conditional_w1w2 = np.mean(np.array(u_hat_conditional_w1w2))
    V_u_hat_conditional_w1w2 = np.mean(np.array(V_u_hat_conditional_w1w2))
    V_non_par_u_hat = np.mean(np.array(V_non_par_u_hat))
    
    #Compute information criterion
    k = len(theta0)
    LL = LL*-1 #Flip for the non-negative likelihood value. 
    AIC = 2*k - 2*LL
    BIC = k*np.log(len(y)) - 2*LL
    final_theta = np.concatenate([theta[:-3], np.array([None, theta[-3], None, None, None, None]), theta[-2:]]).reshape(-1,1)
    final_sterr = np.concatenate([sterr[:-3], np.array([None, sterr[-3], None, None, None, None]), sterr[-2:]]).reshape(-1,1)
    return LL, AIC, BIC, final_theta, final_sterr, u_hat, u_hat_non_par, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat

def global_variable_initializer():
    global base_file_path, eng
    base_file_path = r'D:\Rob\Dropbox (Sydney Uni)\Vine Copula SFA Project'
    eng = matlab.engine.start_matlab()
    eng.addpath('{}/Application'.format(base_file_path), nargout = 0)

def main():
    global_variable_initializer()
    num_integral_sample = 100
    elec_gen_data = pd.read_excel('{}/Application/data.xlsx'.format(base_file_path))
    elec_gen_data.columns = map(str.lower, elec_gen_data.columns)
    elec_gen_data.drop(['unnamed: 0'], axis = 1, inplace = True)
    
    #Inputs x1,x2,x3 are in log form, prices are not
    elec_gen_data_non_log = elec_gen_data.copy(deep = True)
    elec_gen_data[['p1', 'p2', 'p3']] = np.log(elec_gen_data[['p1', 'p2', 'p3']])
    y = elec_gen_data['y'].values.reshape(-1,1)
    x1 = elec_gen_data['x1'].values.reshape(-1,1)
    x2 = elec_gen_data['x2'].values.reshape(-1,1)
    x3 = elec_gen_data['x3'].values.reshape(-1,1)
    p1 = elec_gen_data['p1'].values.reshape(-1,1)
    p2 = elec_gen_data['p2'].values.reshape(-1,1)
    p3 = elec_gen_data['p3'].values.reshape(-1,1)
    n = y.shape[0]
    np.random.seed(10)
    us = stats.norm.ppf((np.random.uniform(size = (num_integral_sample, n))+1)/2, loc = 0, scale = 1) #standard Normal random variables for simulation of the integral.
#    us_ = stats.norm.ppf((np.random.uniform(size = (num_integral_sample, 1))+1)/2, loc = 0, scale = 1) #standard Normal random variables for simulation of the integral.
#    us = np.concatenate([us_ for x in range(n)], axis = 1)
    
    #Summary statistics
    summary_stats = elec_gen_data.describe().T
    summary_stats['median'] = elec_gen_data.median()
    summary_stats.to_csv(r'{}\Application\summary_statistics.csv'.format(base_file_path), index = True)
    elec_gen_data_non_log[['x1', 'x2', 'x3', 'p1', 'p2', 'p3']] = np.exp(elec_gen_data_non_log[['x1', 'x2', 'x3', 'p1', 'p2', 'p3']])
    summary_stats_non_log = elec_gen_data_non_log.describe().T
    
    ### Estimate model parameters ###
    ALS77_LL, ALS77_AIC, ALS77_BIC, ALS77_theta, ALS77_sterr, ALS77_u_hat, ALS77_non_par_u_hat, ALS77_V_u_hat, ALS77_u_hat_conditional_w1w2, ALS77_V_u_hat_conditional_w1w2, ALS77_V_non_par_u_hat = AppEstimate_ALS77(y, x1, x2, x3)
    SL80_LL, SL80_AIC, SL80_BIC, SL80_theta, SL80_sterr, SL80_u_hat, SL80_non_par_u_hat, SL80_V_u_hat, SL80_u_hat_conditional_w1w2, SL80_V_u_hat_conditional_w1w2, SL80_V_non_par_u_hat = AppEstimate_SL80(y, x1, x2, x3, p1, p2, p3)
    APS16_LL, APS16_AIC, APS16_BIC, APS16_theta, APS16_sterr, APS16_u_hat, APS16_non_par_u_hat, APS16_V_u_hat, APS16_u_hat_conditional_w1w2, APS16_V_u_hat_conditional_w1w2, APS16_V_non_par_u_hat = AppEstimate_APS16(y, x1, x2, x3, p1, p2, p3)   
    APS3A_LL, APS3A_AIC, APS3A_BIC, APS3A_theta, APS3A_sterr, APS3A_u_hat, APS3A_non_par_u_hat, APS3A_V_u_hat, APS3A_u_hat_conditional_w1w2, APS3A_V_u_hat_conditional_w1w2, APS3A_V_non_par_u_hat = AppEstimate_APS3A(y, x1, x2, x3, p1, p2, p3, us) 
    APS3B_LL, APS3B_AIC, APS3B_BIC, APS3B_theta, APS3B_sterr, APS3B_u_hat, APS3B_non_par_u_hat, APS3B_V_u_hat, APS3B_u_hat_conditional_w1w2, APS3B_V_u_hat_conditional_w1w2, APS3B_V_non_par_u_hat = AppEstimate_APS3B(y, x1, x2, x3, p1, p2, p3, us) 
    Vine_APS3A_LL, Vine_APS3A_AIC, Vine_APS3A_BIC, Vine_APS3A_theta, Vine_APS3A_sterr, Vine_APS3A_u_hat, Vine_APS3A_non_par_u_hat, Vine_APS3A_V_u_hat, Vine_APS3A_u_hat_conditional_w1w2, Vine_APS3A_V_u_hat_conditional_w1w2, Vine_APS3A_V_non_par_u_hat = AppEstimate_Vine_APS3A(y, x1, x2, x3, p1, p2, p3, us) 
    Vine_APS3B_LL, Vine_APS3B_AIC, Vine_APS3B_BIC, Vine_APS3B_theta, Vine_APS3B_sterr, Vine_APS3B_u_hat, Vine_APS3B_non_par_u_hat, Vine_APS3B_V_u_hat, Vine_APS3B_u_hat_conditional_w1w2, Vine_APS3B_V_u_hat_conditional_w1w2, Vine_APS3B_V_non_par_u_hat = AppEstimate_Vine_APS3B(y, x1, x2, x3, p1, p2, p3, us) 
    
    #concatenate parameter eastimation results
    parameter_estimation_results = pd.DataFrame(columns = ['ALS77', 'ALS77', 'SL80', 'SL80', 'APS16', 'APS16', 'APS3A', 'APS3A', 'APS3B', 'APS3B', 'Vine APS2A', 'Vine APS2A', 'Vine APS2B', 'Vine APS2B'], 
                                        index = [' ', '$\alpha$', '$\beta_{1}$', '$\beta_{2}$', '$\beta_{3}$', '$\mu_{1}', '$\mu_{2}$', '$\sigma^{2}_{u}$', '$\sigma^{2}_{v}$', 
                                                 '$\sigma^{2}_{w_{1}}$', '$\sigma^{2}_{w_{2}}$', '\sigma_{w_{1},w_{2}}}', '$\rho_{w}_{1},w_{2}}$', '\sigma_{u,w_{1}}', '\sigma_{u,w_{2}}', '\sigma_{v,w_{1}}', '\sigma_{v,w_{2}}', '$\theta_{12}$', '$\theta_{13}$'])
    l = []
    l += int((len(parameter_estimation_results.columns)/2))*['Est', 'Std Err']
    parameter_estimation_results.loc[' ',:] = l

    theta_frame = np.concatenate([ALS77_theta, ALS77_sterr, SL80_theta, SL80_sterr, APS16_theta, APS16_sterr, APS3A_theta, APS3A_sterr, APS3B_theta, APS3B_sterr, 
                                  Vine_APS3A_theta, Vine_APS3A_sterr, Vine_APS3B_theta, Vine_APS3B_sterr], axis = 1)
    parameter_estimation_results.iloc[1:,:] = theta_frame
    
    #Likelihood and information criterion
    AIC_frame = np.array([ALS77_AIC, SL80_AIC, APS16_AIC, APS3A_AIC, APS3B_AIC, Vine_APS3A_AIC, Vine_APS3B_AIC])
    BIC_frame = np.array([ALS77_BIC, SL80_BIC, APS16_BIC, APS3A_BIC, APS3B_BIC, Vine_APS3A_BIC, Vine_APS3B_BIC])
    LL_frame = np.array([ALS77_LL, SL80_LL, APS16_LL, APS3A_LL, APS3B_LL, Vine_APS3A_LL, Vine_APS3B_LL])
        
    model_info_results = pd.DataFrame(columns = ['ALS77', 'SL80', 'APS16', 'APS3A', 'APS3B', 'Vine APS2A', 'Vine APS2B'], index = ['LL', 'AIC', 'BIC'])
    model_info_results.loc['LL', :] = LL_frame
    model_info_results.loc['AIC', :] = AIC_frame
    model_info_results.loc['BIC', :] = BIC_frame
    
    #Technical inefficiency prediction
    u_hat_frame = np.array([ALS77_u_hat, SL80_u_hat, APS16_u_hat, APS3A_u_hat, APS3B_u_hat, Vine_APS3A_u_hat, Vine_APS3B_u_hat])
    V_u_hat_frame = np.array([ALS77_V_u_hat, SL80_V_u_hat, APS16_V_u_hat, APS3A_V_u_hat, APS3B_V_u_hat, Vine_APS3A_V_u_hat, Vine_APS3B_V_u_hat])
    non_par_u_hat = np.array([ALS77_non_par_u_hat, SL80_non_par_u_hat, APS16_non_par_u_hat, APS3A_non_par_u_hat, APS3B_non_par_u_hat, Vine_APS3A_non_par_u_hat, Vine_APS3B_non_par_u_hat])
    u_hat_conditional_w1w2_frame = np.array([ALS77_u_hat_conditional_w1w2, SL80_u_hat_conditional_w1w2, APS16_u_hat_conditional_w1w2, APS3A_u_hat_conditional_w1w2, APS3B_u_hat_conditional_w1w2, Vine_APS3A_u_hat_conditional_w1w2, Vine_APS3B_u_hat_conditional_w1w2])
    V_u_hat_conditional_w1w2_frame = np.array([ALS77_V_u_hat_conditional_w1w2, SL80_V_u_hat_conditional_w1w2, APS16_V_u_hat_conditional_w1w2, APS3A_V_u_hat_conditional_w1w2, APS3B_V_u_hat_conditional_w1w2, Vine_APS3A_V_u_hat_conditional_w1w2, Vine_APS3B_V_u_hat_conditional_w1w2])
    V_non_par_u_hat_frame = np.array([ALS77_V_non_par_u_hat, SL80_V_non_par_u_hat, APS16_V_non_par_u_hat, APS3A_V_non_par_u_hat, APS3B_V_non_par_u_hat, Vine_APS3A_V_u_hat_conditional_w1w2, Vine_APS3B_V_non_par_u_hat])

    u_hat_results = pd.DataFrame(columns = ['ALS77', 'SL80', 'APS16', 'APS3A', 'APS3B', 'Vine APS2A', 'Vine APS2B'], 
                                 index = ['$E[u|\eps]$', '$V[u|\eps]$', '$\tilde{E}[u|\eps]$', '$\tilde{V}[u|\eps]$', '$E[u|\mathbf{\omega}, \eps]$', '$V[u|\mathbf{\omega}, \eps]$'])
    u_hat_results.loc['$E[u|\eps]$', :] = u_hat_frame
    u_hat_results.loc['$V[u|\eps]$', :] = V_u_hat_frame
    u_hat_results.loc['$\tilde{E}[u|\eps]$', :] = non_par_u_hat
    u_hat_results.loc['$\tilde{V}[u|\eps]$', :] = V_non_par_u_hat_frame
    u_hat_results.loc['$E[u|\mathbf{\omega}, \eps]$', :] = u_hat_conditional_w1w2_frame
    u_hat_results.loc['$V[u|\mathbf{\omega}, \eps]$', :] = V_u_hat_conditional_w1w2_frame

    #Export results
    parameter_estimation_results.to_csv(r'{}\Application\parameter_estimation_results.csv'.format(base_file_path), index = True)
    model_info_results.to_csv(r'{}\Application\model_info_results.csv'.format(base_file_path), index = True)
    u_hat_results.to_csv(r'{}\Application\u_hat_results.csv'.format(base_file_path), index = True)
    eng.quit()
    
if __name__ == '__main__':  
    main()