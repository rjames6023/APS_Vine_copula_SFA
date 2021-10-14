function [logMLE, theta, sterr, u_hat, V_u_hat, non_par_u_hat, u_hat_conditional_w2w3, V_u_hat_conditional_w2w3, V_non_par_u_hat] = nested_AppLoglikelihood_SL80(y, x1, x2, x3, p1, p2, p3, theta0)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE] = fminunc(@AppLoglikelihood_SL80, theta0, Options); %Maximise the log-likelihood
%     rng(10) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display','off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_SL80,'options',Options);
%     [theta, logMLE] = run(gs, problem);
    
    %get the covariance matrix back from the Cholesky decomposition
    theta_end = length(theta);
    theta(theta_end-2) = exp(theta(theta_end-2));
    Sigma = CholeskiToMatrix(theta(5:10));
    theta(5:10)= Sigma(itril(size(Sigma)))';

    %Estimation of standard errors 
    delta = 1e-6;
    grad = zeros(length(y), length(theta));
    for i=1:length(theta)
        theta1 = theta;
        theta1(i) = theta(i) + delta;
        grad(:,i) = (AppLogDen_SL80(theta1) - AppLogDen_SL80(theta))/delta; 
    end
    OPG = grad'*grad; %Outer product of the gradient 
    sterr = sqrt(diag(inv(OPG))); 
    
    %Prediction of technical inefficiency
    theta_end = length(theta);
    alpha = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    beta3 = theta(4);
    mu2 = theta(theta_end-1);
    mu3 = theta(theta_end);
    
    Sigma = zeros(3, 3);
    Sigma(itril(size(Sigma))) = theta(5:10); %insert the lower triangular values of the covariance matrix
    Sigma = Sigma + Sigma' - diag(diag(Sigma)); %compute the upper triangular values of the covariance matrix and subtract the diagonal to account for double sum
    sigma2v = theta(theta_end-2); %variance of the random noise term
    sigma2u = Sigma(1,1);
    
    obs_eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3; %composed errors from the production function equation (i.e residuals from the production function)
    lambda = sqrt(sigma2u/sigma2v);
    sigma = sqrt(sigma2u+sigma2v);
    sig_star = sqrt(sigma2u*sigma2v/(sigma^2));
    u_hat = sig_star*(((normpdf(lambda.*obs_eps./sigma, 0, 1))./(1-normcdf(lambda.*obs_eps./sigma))) - ((lambda.*obs_eps)./sigma)); %Conditional distribution of u given eps. 
    V_u_hat = sig_star^2*(1+normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma)).*lambda.*obs_eps./sigma-(normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma))).^2);
    
        %Prediction of technical inefficiency conditional on w1 and w2
    S_kernel = 10000; %number of simulated draws for evaluation of the expectation from joint distribution of (u,w21,w2)
    n = length(y);
    
            %Observed variables
    B2 = p2 - p1 +log(beta1) - log(beta2);
    B3 = p3 - p1 +log(beta1) - log(beta3);
    obs_w2 = x1 - x2 - B2;
    obs_w3 = x1 - x3 - B3; 
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);
    rep_obs_w2 = reshape(repelem(obs_w2, S_kernel), S_kernel, n);
    rep_obs_w3 = reshape(repelem(obs_w3, S_kernel), S_kernel, n);
    
        %SL80 implies a specific gaussian copula, see Appendix 1 of APS2020
            %Correlation matrtix 1
    R1 = [1, Sigma(2,1)/sqrt(Sigma(1,1)*Sigma(2,2)), Sigma(3,1)/sqrt(Sigma(1,1)*Sigma(3,3)); Sigma(1,2)/sqrt(Sigma(1,1)*Sigma(2,2)), 1, Sigma(3,2)/sqrt(Sigma(2,2)*Sigma(3,3)); Sigma(1,3)/sqrt(Sigma(1,1)*Sigma(3,3)), Sigma(2,3)/sqrt(Sigma(2,2)*Sigma(3,3)), 1];
    R2 = [1, -(Sigma(2,1)/sqrt(Sigma(1,1)*Sigma(2,2))), -(Sigma(3,1)/sqrt(Sigma(1,1)*Sigma(3,3))); -(Sigma(1,2)/sqrt(Sigma(1,1)*Sigma(2,2))), 1, Sigma(3,2)/sqrt(Sigma(2,2)*Sigma(3,3)); -(Sigma(1,3)/sqrt(Sigma(1,1)*Sigma(3,3))), Sigma(2,3)/sqrt(Sigma(2,2)*Sigma(3,3)), 1];
    SL80_copula_samples = 0.5*copularnd('Gaussian', R1, S_kernel) + 0.5*copularnd('Gaussian', R2, S_kernel);
    
    rng(10) %Set seed
    simulated_v = normrnd(0, sqrt(sigma2v), S_kernel, 1);
    simulated_u = sqrt(Sigma(1,1))*norminv((SL80_copula_samples(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_w2 = norminv(SL80_copula_samples(:,2), mu2, sqrt(Sigma(2,2)));
    simulated_w3 = norminv(SL80_copula_samples(:,3), mu3, sqrt(Sigma(3,3)));
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)    
    
        %Bandwidth information for each conditioning variable
    h = 1.06*n^(-1/5)*[max(std(simulated_eps), iqr(simulated_eps)/1.34) max(std(simulated_w2), iqr(simulated_w2)/1.34) max(std(simulated_w3), iqr(simulated_w3)/1.34)];
 
        %Compute kernel estimates for E[u|eps]
    kernel_regression_results1 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        kernel_product = eps_kernel;
        kernel_regression_results1(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
    end
    non_par_u_hat = kernel_regression_results1;
    
        %Compute kernel estimates for E[u|eps, w1, w2]
    kernel_regression_results = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        w2_kernel = normpdf((simulated_w2 - rep_obs_w2(:,i))./h(2));
        w3_kernel = normpdf((simulated_w3 - rep_obs_w3(:,i))./h(3));
        kernel_product = eps_kernel.*w2_kernel.*w3_kernel;
        kernel_regression_results(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
    end
    u_hat_conditional_w2w3 = kernel_regression_results;
    
        %V[u|eps, w1, w2] (conditional variance of the random variable u)
    kernel_regression_results2 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        w2_kernel = normpdf((simulated_w2 - rep_obs_w2(:,i))./h(2));
        w3_kernel = normpdf((simulated_w3 - rep_obs_w3(:,i))./h(3));
        kernel_product = eps_kernel.*w2_kernel.*w3_kernel;
        kernel_regression_results2(i,1) = sum(kernel_product.*(simulated_u.^2))/sum(kernel_product);
    end
    V_u_hat_conditional_w2w3 = kernel_regression_results2 - (kernel_regression_results.^2);
    
    kernel_regression_results2 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        kernel_product = eps_kernel;
        kernel_regression_results2(i,1) = sum(kernel_product.*(simulated_u.^2))/sum(kernel_product);
    end
    V_non_par_u_hat = kernel_regression_results2 - (non_par_u_hat.^2);
    
        %kernel variance estimation (variance of the residuals (CEF)
%     leave_one_out_fitted_values = zeros(n,1);
%     for i = 1:S_kernel
%         leave_one_out_simulated_eps = simulated_eps;
%         leave_one_out_simulated_eps(i) = [];
%         leave_one_out_simulated_w2 = simulated_w2;
%         leave_one_out_simulated_w2(i) = [];
%         leave_one_out_simulated_w3 = simulated_w2;
%         leave_one_out_simulated_w3(i) = [];
%         leave_one_out_simulated_u = simulated_u;
%         leave_one_out_simulated_u(i) = [];
%         
%         leave_one_out_eps_kernel = normpdf((leave_one_out_simulated_eps - simulated_eps(i))./h(1));
%         leave_one_out_w2_kernel = normpdf((leave_one_out_simulated_w2 - simulated_w2(i))./h(2));
%         leave_one_out_w3_kernel = normpdf((leave_one_out_simulated_w3 - simulated_w3(i))./h(3));
%         leave_one_out_kernel_product = leave_one_out_eps_kernel.*leave_one_out_w2_kernel.*leave_one_out_w3_kernel;
%         leave_one_out_fitted_values(i,1) = sum(leave_one_out_kernel_product.*leave_one_out_simulated_u)/sum(leave_one_out_kernel_product);
%     end
%     leave_one_out_residuals = (simulated_u - leave_one_out_fitted_values).^2;
%     
%             %kernel variance estimation
%     h_var = 1.06*111^(-1/5)*[max(std(simulated_eps), iqr(simulated_eps)/1.34) max(std(simulated_w2), iqr(simulated_w2)/1.34) max(std(simulated_w3), iqr(simulated_w3)/1.34)];
%     kernel_regression_variance_results = zeros(n,1);
%     for i = 1:S_kernel
%         eps_kernel = normpdf((simulated_eps - simulated_eps(i))./h_var(1));
%         w2_kernel = normpdf((simulated_w2 - simulated_w2(i))./h_var(2));
%         w3_kernel = normpdf((simulated_w3 - simulated_w2(i))./h_var(3));
%         kernel_product = eps_kernel.*w2_kernel.*w3_kernel;
%         kernel_regression_variance_results(i,1) = sum(kernel_product.*leave_one_out_residuals)/sum(kernel_product);
%     end
%     V_e_conditional_w2w3 = kernel_regression_variance_results;
    
    function logL = AppLoglikelihood_SL80(coefs)
        % tranform parameters back true range
        theta_end = length(coefs);
        coefs(theta_end-2) = exp(coefs(theta_end-2)); %transform the sigma2v back to correct range
        Sigma = CholeskiToMatrix(coefs(5:10)); %get the covariance matrix back from the Cholesky decomposition
        coefs(5:10)= Sigma(itril(size(Sigma)))'; %grab the (colum-wise) flattened elements of the lower triangular covariance matrix
        
        logDen = AppLogDen_SL80(coefs);
        logL = -sum(logDen);
    end
    
	function logDen = AppLogDen_SL80(coefs) 
        theta_end = length(coefs);
        alpha = coefs(1);
        beta1 = coefs(2);
        beta2 = coefs(3);
        beta3 = coefs(4);

        Sigma = zeros(3, 3);
        Sigma(itril(size(Sigma))) = coefs(5:10); %insert the lower triangular values of the covariance matrix
        Sigma = Sigma + Sigma' - diag(diag(Sigma)); %compute the upper triangular values of the covariance matrix and subtract the diagonal to account for double sum
        sigma2v = coefs(theta_end-2); %variance of the random noise term
        mu = [0, coefs(theta_end-1:theta_end)']; %mean vector for MVN distribution of allocative error terms

        G = Sigma;
        G(1,1) = G(1,1) + sigma2v; %eq. (A.3) from SL80
        Gstar = G;
        Gstar(1, 2:3) = - G(1, 2:3);
        Gstar(2:3, 1) = - G(2:3, 1);
        
        eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3; %composed errors from the production function equation (i.e residuals from the production function)
        B2 = p2 - p1 +log(beta1)-log(beta2); %from eq.6 SL80
        B3 = p3 - p1 +log(beta1)-log(beta3); %from eq.6 SL80
        w2 = x1 - x2 - B2; %1st allocative error term
        w3 = x1 - x3 - B3; %2nd allocative error term
        w = [w2 w3]; %vector of allocative error terms
        all = [eps w];
        r = beta1 + beta2 + beta3;
        InvSigma = inv(Sigma);
      
        A = (eps + sigma2v*w*InvSigma(1, 2:3)')/sqrt(sigma2v)*sqrt(det(Sigma)/det(G)); %eq. (A.5) from SL80
        Astar = (eps - sigma2v*w*InvSigma(1, 2:3)')/sqrt(sigma2v)*sqrt(det(Sigma)/det(G)); %eq. (A.6) from SL80
        logDen = log(r) + log(normcdf(-A,0,1).*mvnpdf(all, mu, G) + normcdf(-Astar,0,1).*mvnpdf(all, mu, Gstar));
    end
end










