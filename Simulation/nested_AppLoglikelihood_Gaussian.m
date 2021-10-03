function [theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2] = nested_AppLoglikelihood_Gaussian(y, x1, x2, z1, z2, us, theta0, random_seed)
    %dlmwrite('log_test_file.txt','start','-append')
    rng(random_seed) %Set seed
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE, grad, hessian] = fminunc(@AppLoglikelihood_Gaussian, theta0, Options);
%     rng(random_seed) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_Gaussian,'options',Options, ...
%                                  'lb',[0, 0, 0, log(0.5), log(0.5), log(0.5), log(0.5), 0.5, -1, -1], ...
%                                  'ub',[1, 1, 1, log(1.5), log(1.5), log(1.5), log(1.5), 1.5, 1, 1]);
%     [theta, logMLE] = run(gs, problem);
    
    %Estimate standard errors 
     theta(4:7) = exp(theta(4:7)); %Transform parameters back to the true range
    %Trim the final 2 columns from the hessian (remove parameter rho and pw1w2)
%     [hessian_N, hessian_P] = size(hessian);
%     hessian = hessian(:,1:hessian_P-2);
    
    delta = 1e-6;
    grad_contributions = zeros(length(y), length(theta)-2);
    for i=1:length(theta)-2
        theta1 = theta;
        theta1 = [theta1', theta(end-1), theta(end)]; %Add back rho and pw1w2 for density estimation
        theta1(i) = theta(i) + delta;
        grad_contributions(:,i) = (AppLogDen_Gaussian(theta1)-AppLogDen_Gaussian(theta))/delta; 
    end
    OPG = grad_contributions'*grad_contributions; 
    sterr = sqrt(diag(inv(OPG)));
    
    %Prediction of technical inefficiency
    alpha = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    sigmav2 = theta(4);
    sigmau2 = theta(5);

    obs_eps = y - alpha - x1*beta1 - x2*beta2; %composed errors from the production function equation (i.e residuals from the production function)
    lambda = sqrt(sigmau2/sigmav2);
    sigma = sqrt(sigmau2+sigmav2);
    sig_star = sqrt(sigmau2*sigmav2/(sigma^2));
    u_hat = sig_star*(((normpdf(lambda.*obs_eps./sigma, 0, 1))./(1-normcdf(lambda.*obs_eps./sigma))) - ((lambda.*obs_eps)./sigma)); %Conditional distribution of u given eps. 
    
    %Prediction of technical inefficiency using information from the joint distribution
    sigma2w1 = theta(6);
    sigma2w2 = theta(7);
    gamma = theta(8);
    rho = theta(9);
    pw1w2 = theta(10);
    S_kernel = 10000; %number of simulated draws for evaluation of the expectation from joint distribution of (u,w21,w2)
    n = length(y);
    Sigma=[1, rho, rho; rho,1,pw1w2; rho, pw1w2,1];

    %Observed variables
    obs_w1 = x1 - z1*gamma;
    obs_w2 = x2 - z2*gamma;
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);
    rep_obs_w1 = reshape(repelem(obs_w1, S_kernel), S_kernel, n);
    rep_obs_w2 = reshape(repelem(obs_w2, S_kernel), S_kernel, n);

    %Simulated variables
    simulated_errors = copularnd('Gaussian',Sigma,S_kernel); %S_kernelx3 matrix of simulated error terms from gaussian copula (u,w1,w2). Returned as U(0,1) numbers
    simulated_v = normrnd(0, sqrt(sigmav2), S_kernel, 1);
    simulated_u = sqrt(sigmau2)*norminv((simulated_errors(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_w1 = norminv(simulated_errors(:,2), 0, sqrt(sigma2w1));
    simulated_w2 = norminv(simulated_errors(:,3), 0, sqrt(sigma2w2));
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)

    %Bandwidth information for each conditioning variable
    h = 1.06*n^(-1/5)*[max(std(obs_eps), iqr(obs_eps)/1.34) max(std(obs_w1), iqr(obs_w1)/1.34) max(std(obs_w2), iqr(obs_w2)/1.34)];

     %Compute kernel estimates for E[u|eps, w1, w2]
    kernel_regression_results1 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        kernel_product = eps_kernel;
        kernel_regression_results1(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
    end
    non_par_u_hat = kernel_regression_results1;
    
    %Compute kernel estimates for E[u|eps, w1, w2]
    kernel_regression_results2 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        w1_kernel = normpdf((simulated_w1 - rep_obs_w1(:,i))./h(2));
        w2_kernel = normpdf((simulated_w2 - rep_obs_w2(:,i))./h(3));
        kernel_product = eps_kernel.*w1_kernel.*w2_kernel;
        kernel_regression_results2(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
    end
    u_hat_conditional_w1w2 = kernel_regression_results2;
    
    function logL = AppLoglikelihood_Gaussian(coefs)
        % tranform parameters back true range
        coefs(4:7) =  exp(coefs(4:7));
        logDen = AppLogDen_Gaussian(coefs);
        logL = -sum(logDen);
    end

    function logDen = AppLogDen_Gaussian(coefs)        
        N = length(y); %n obs
        [S,P] = size(us); %number of simulated draws for evaluation of the integral
        
        % obtain the log likelihood
        alpha = coefs(1);
        beta1 = coefs(2);
        beta2 = coefs(3);
        sigma2v = coefs(4); 
        sigma2u = coefs(5);
        sigma2w1 = coefs(6);
        sigma2w2 = coefs(7);
        gamma = coefs(8);
        rho = coefs(9);
        pw1w2 = coefs(10);

        eps = y-alpha-x1*beta1-x2*beta2; %composed errors from the production function equation (i.e residuals from the production function)
        w1 = x1 - z1*gamma; %re-arranged endogenous regressor equations p.11
        w2 = x2 - z2*gamma; %re-arranged endogenous regressor equations p.11

        DenW1 = normpdf(w1, 0, sqrt(sigma2w1)); %marginal density of allocative inefficiency terms
        DenW2 = normpdf(w2, 0, sqrt(sigma2w2)); %marginal density of allocative inefficiency terms
        
        eps_SxN = reshape(repelem(eps, S), S, N);
        us_SxN = sqrt(sigma2u)*us;
        CdfUs = 2*(normcdf(sqrt(sigma2u)*us, 0, sqrt(sigma2u)) -0.5);
        EpsPlusUs = eps_SxN + us_SxN;
        DenEpsPlusUs = normpdf(EpsPlusUs, 0, sqrt(sigma2v));
        %pd2 = makedist('HalfNormal','mu',0,'sigma',sqrt(sigma2u));
        %CdfUs = cdf(pd2,us);
        
        Sigma=[1, rho, rho; rho,1,pw1w2; rho, pw1w2,1]; %Gaussian copula correlation matrix. Correlation between u and w_{i} = rho. Correlation between w_{i} and w_{j} = rho
        %Check if Sigma is positive definite
        [R,flag] = chol(Sigma);
        if flag ~= 0
            logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if Sigma is not positve definite
        else
            %Evaluate the integral via simulation (to integrate out u from eps+u)
            simulated_copula_pdfs = zeros(S,N);
            w1_rep = reshape(repelem(w1, S),S, N);
            w2_rep = reshape(repelem(w2, S),S, N);
                %Compute the CDF (standard normal) for repeated allocative inefficiency terms
            CDF_w1_rep = normcdf(w1_rep, 0, sqrt(sigma2w2));
            CDF_w2_rep = normcdf(w2_rep, 0, sqrt(sigma2w2));
            for j = 1:N
                c123 = copulapdf('Gaussian',[CdfUs(:,j),CDF_w1_rep(:,j),CDF_w2_rep(:,j)],Sigma);
                simulated_copula_pdfs(:,j) = c123;
            end
            Integral = mean(simulated_copula_pdfs.*DenEpsPlusUs)'; %Evaluation of the integral over S simulated samples. Column-wise mean.
            DenAll = DenW1.*DenW2.*Integral; %joint desnity. product of marginal density of w_{1} and w_{2} and the joint density f(\epsilon, w)
            logDen = log(DenAll);
        end
    end 
end