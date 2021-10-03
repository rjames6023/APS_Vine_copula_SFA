function [logMLE, theta, sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_u_hat_conditional_w1w2, V_non_par_u_hat] = nested_AppLoglikelihood_APS16(y, x1, x2, x3, p1, p2, p3, theta0)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE] = fminunc(@AppLoglikelihood_APS16, theta0, Options); %Maximise the log-likelihood
%     rng(10) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display','off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_APS16,'options',Options);
%     [theta, logMLE] = run(gs, problem);

    %get the covariance matrix back from the Cholesky decomposition
    theta_end = length(theta);
    theta(theta_end) = exp(theta(theta_end));
    Sigma = CholeskiToMatrix(theta(5:10));
    theta(5:10)= Sigma(itril(size(Sigma)))';
    
    %Estimation of standard errors 
    delta = 1e-6;
    grad = zeros(length(y), length(theta));
    for i=1:length(theta)
        theta1 = theta;
        theta1(i) = theta(i) + delta;
        grad(:,i) = (AppLogDen_APS16(theta1) - AppLogDen_APS16(theta))/delta; 
    end
    OPG = grad'*grad; %Outer product of the gradient 
    sterr = sqrt(diag(inv(OPG))); 

    %Prediction of technical inefficiency
    theta_end = length(theta);
    alpha = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    beta3 = theta(4);
    sigma2u = theta(theta_end);

    Sigma = zeros(3,3);
    Sigma(itril(size(Sigma))) = theta(5:10); %insert the lower triangular values of the covariance matrix
    Sigma = Sigma + Sigma'-diag(diag(Sigma)); %compute the upper triangular values of the covariance matrix and subtract the diagonal to account for double sum
    G = Sigma; 
    covvw = G(2:3, 1)'; %from eq.(13d)
    covww = G(2:3, 2:3); %from eq.(13d)
    covwv = covvw';
    sigma2v = Sigma(1,1);
    sigma2c = sigma2v - covvw*inv(covww)*covwv;
    
    B2 = p2 - p1 +log(beta1)-log(beta2);
    B3 = p3 - p1 +log(beta1)-log(beta3);
    w2 = x1 - x2 - B2; %from eq.6 SL80
    w3 = x1 - x3 - B3; %from eq.6 SL80
    w = [w2, w3]; %vector of allocative error terms
        
    uci = covvw*inv(covww)*w'; %from eq.(13d)
     
    obs_eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3 - uci'; %composed errors from the production function equation (i.e residuals from the production function)
    lambda = sqrt(sigma2u/sigma2c);
    sigma = sqrt(sigma2u+sigma2v);
    sig_star = sqrt(sigma2u*sigma2v/(sigma^2));
    u_hat = sig_star*(((normpdf(lambda.*obs_eps./sigma, 0, 1))./(1-normcdf(lambda.*obs_eps./sigma))) - ((lambda.*obs_eps)./sigma)); %Conditional distribution of u given eps. 
    V_u_hat = sig_star^2*(1+normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma)).*lambda.*obs_eps./sigma-(normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma))).^2);
    
    %Prediction of technical inefficiency conditional on w1 and w2
        %w1 and w2 are correlated with v and can be informative abou E[u|eps], see APS16 section 4.4
    eps_tilde = obs_eps - uci;
    h_ = (lambda/sigma)*eps_tilde;
    sig2_star = (sigma2u* sigma2c)/(sigma^2);
    hazard = normpdf(h_)/(1 - normcdf(h_));
    u_hat_conditional_w1w2 = sqrt(sig2_star)*(hazard - h_);
    V_u_hat_conditional_w1w2 = sig2_star*(1+normpdf(h_)./(1-normcdf(h_)).*h_-(normpdf(h_)./(1-normcdf(h_))).^2);
    
        %Prediction of technical inefficiency conditional on w1 and w2
    S_kernel = 10000; %number of simulated draws for evaluation of the expectation from joint distribution of (u,w21,w2)
    n = length(y);
    
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);
    rep_obs_w2 = reshape(repelem(w2, S_kernel), S_kernel, n);
    rep_obs_w3 = reshape(repelem(w3, S_kernel), S_kernel, n);
    
    rng(10) %Set seed
    APS16_samples = mvnrnd([0, 0, 0], Sigma, S_kernel);
    simulated_v = APS16_samples(:,1);
    simulated_u = sqrt(sigma2u)*norminv((rand(S_kernel,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)  
    
    %Bandwidth information for each conditioning variable
    h = 1.06*n^(-1/5)*[max(std(obs_eps), iqr(obs_eps)/1.34)];
   
    %Compute kernel estimates for E[u|eps]
    kernel_regression_results1 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        kernel_product = eps_kernel;
        kernel_regression_results1(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
    end
    non_par_u_hat = kernel_regression_results1;
    
        kernel_regression_results2 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        kernel_product = eps_kernel;
        kernel_regression_results2(i,1) = sum(kernel_product.*(simulated_u.^2))/sum(kernel_product);
    end
    V_non_par_u_hat = kernel_regression_results2 - (non_par_u_hat.^2);
    
    function logL = AppLoglikelihood_APS16(coefs)
        % tranform parameters back true range
        theta_end = length(coefs);
        coefs(theta_end) = exp(coefs(theta_end)); %transform the sigma2v back to correct range
        Sigma = CholeskiToMatrix(coefs(5:10)); %get the covariance matrix back from the Cholesky decomposition
        coefs(5:10)= Sigma(itril(size(Sigma)))'; %grab the (colum-wise) flattened elements of the lower triangular covariance matrix
        
        logDen = AppLogDen_APS16(coefs);
        logL = -sum(logDen);
    end
    
	function logDen = AppLogDen_APS16(coefs) 
        theta_end = length(coefs);
        alpha = coefs(1);
        beta1 = coefs(2);
        beta2 = coefs(3);
        beta3 = coefs(4);
        
        %Construct covariance matrix information
        Sigma = zeros(3,3);
        Sigma(itril(size(Sigma))) = coefs(5:10); %insert the lower triangular values of the covariance matrix
        Sigma = Sigma + Sigma'-diag(diag(Sigma)); %compute the upper triangular values of the covariance matrix and subtract the diagonal to account for double sum
        G = Sigma; 
        sigma2u = coefs(theta_end);
        sigma2v = Sigma(1,1);
        SigmaWW = Sigma(2:3, 2:3);
        
        covvw = G(2:3, 1)'; %from eq.(13d)
        covww = G(2:3, 2:3); %from eq.(13d)
        covwv = covvw';
        
        sigma2c = sigma2v - covvw*inv(covww)*covwv;
        sigma2 = sigma2u+sigma2c;
        lambda = sqrt(sigma2u/sigma2c);
        sigma = sqrt(sigma2);
        
        B2 = p2 - p1 +log(beta1)-log(beta2);
        B3 = p3 - p1 +log(beta1)-log(beta3);
        w2 = x1 - x2 - B2; %from eq.6 SL80
        w3 = x1 - x3 - B3; %from eq.6 SL80
        w = [w2, w3]; %vector of allocative error terms
        
        uci = covvw*inv(covww)*w'; %from eq.(13d)
     	eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3 - uci'; %see e.g. eq.(13a)
        
        %Compute the log density
        r = beta1 + beta2 + beta3;
        Den = 2/sigma*normpdf(eps/sigma,0,1).*normcdf(-lambda*eps/sigma,0,1); %eq. 8 ALS77 using eps - uci from APS16, equal to eq.(13b) from APS16
        logDen_Tech = log(Den);
        logDen_Alloc = log(mvnpdf(w, 0, SigmaWW)); % eq.(13c) APS16
        logDen = log(r) + logDen_Tech + logDen_Alloc; 
    end
    
end