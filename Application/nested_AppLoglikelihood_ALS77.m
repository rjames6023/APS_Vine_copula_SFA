function [logMLE, theta, sterr, u_hat, non_par_u_hat, V_u_hat, u_hat_conditional_w1w2, V_non_par_u_hat] = nested_AppLoglikelihood_ALS77(y, x1, x2, x3, theta0)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE] = fminunc(@AppLoglikelihood_ALS77, theta0, Options); %Maximise the log-likelihood
%     rng(10) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display','off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_ALS77,'options',Options);
%     [theta, logMLE] = run(gs, problem);
    
    %Transform parameter back to true range
    theta(5:6) = exp(theta(5:6)); 
    
    %Estimation of standard errors 
    delta = 1e-6;
    grad = zeros(length(y), length(theta));
    for i=1:length(theta)
        theta1 = theta;
        theta1(i) = theta(i) + delta;
        grad(:,i) = (AppLogDen_ALS77(theta1)-AppLogDen_ALS77(theta))/delta; 
    end
    OPG = grad'*grad; %Outer product of the gradient 
    sterr = sqrt(diag(inv(OPG)));  
    
    %Prediction of technical inefficiency
    alpha = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    beta3 = theta(4);
    sigma2v = theta(5);
    sigma2u = theta(6);

    obs_eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3; %composed errors from the production function equation (i.e residuals from the production function)
    lambda = sqrt(sigma2u/sigma2v);
    sigma = sqrt(sigma2u+sigma2v);
    sig_star = sqrt(sigma2u*sigma2v/(sigma^2));
    u_hat = sig_star*(((normpdf(lambda.*obs_eps./sigma, 0, 1))./(1-normcdf(lambda.*obs_eps./sigma))) - ((lambda.*obs_eps)./sigma)); %Conditional distribution of u given eps. 
    V_u_hat = sig_star^2*(1+normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma)).*lambda.*obs_eps./sigma-(normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma))).^2);
    
    S_kernel = 10000; %number of simulated draws
    n = length(y);
    rng(10) %Set seed
    simulated_v = normrnd(0, sqrt(sigma2v), S_kernel, 1);
    simulated_u = sqrt(sigma2u)*norminv((rand(S_kernel,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)   
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);
    
        %Compute kernel estimates for E[u|eps]
            %Bandwidth information for each conditioning variable
    h = 1.06*n^(-1/5)*[max(std(obs_eps), iqr(obs_eps)/1.34)];    
    
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
    
    u_hat_conditional_w1w2 = 0;
      
    function logL = AppLoglikelihood_ALS77(coefs)
        % tranform parameters back true range
        coefs(5:6) =  exp(coefs(5:6));
        logDen = AppLogDen_ALS77(coefs);
        logL = -sum(logDen);
    end 

    function logDen = AppLogDen_ALS77(coefs)
         alpha = coefs(1);
         beta1 = coefs(2);
         beta2 = coefs(3);
         beta3 = coefs(4);
         sigma2u = coefs(5);
         sigma2v = coefs(6);

         lambda = sqrt(sigma2u/sigma2v); %Square root of the ratio of variance of technical and random errors.
         sigma2 = sigma2u+sigma2v; %Variance of the composed error term
         sigma = sqrt(sigma2);
         eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3; %Composed errors from the production function equation (i.e residuals)

         Den = 2/sigma.*normpdf(eps/sigma,0,1).*(1-normcdf((lambda*eps)/sigma,0,1)); %Density of the SFA model evaluated at the data
         logDen = log(Den); %Log density
    end
 
end
