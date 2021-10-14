function [logMLE, theta, sterr, u_hat, V_u_hat, non_par_u_hat, u_hat_conditional_w2w3, V_u_hat_conditional_w2w3, V_non_par_u_hat] = nested_AppLoglikelihood_APS3A_unconstrained(y, x1, x2, x3, p1, p2, p3, us, theta0)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE] = fminunc(@AppLoglikelihood_APS3A, theta0, Options); %Maximise the log-likelihood
%      rng default %Random seed for global search problem
%      gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display','off');
%      problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_APS3A,'lb',[-13,0,0,0,1.5,-0.6,0,0,0,0,0,0.01,-1],'ub',[-9,1,2,1,2.2,-0.5,1,1,1,1,1,1,-0.01], 'options',Options);
%      [theta, logMLE] = run(gs, problem);

    %transform parameters back to true range
    theta(7:10) = exp(theta(7:10));
    
    %Estimation of standard errors 
    delta = 1e-6;
    grad = zeros(length(y), length(theta));
    for i=1:length(theta)
        theta1 = theta;
        theta1(i) = theta(i) + delta;
        grad(:,i) = (AppLogDen_APS3A(theta1) - AppLogDen_APS3A(theta))/delta; 
    end
    OPG = grad'*grad; %Outer product of the gradient 
    sterr = sqrt(diag(inv(OPG))); 

    %Prediction of technical inefficiency
    alpha = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    beta3 = theta(4);
    mu2 = theta(5);
    mu3 = theta(6);
    sigma2u = theta(7);
    sigma2v = theta(8);
    sigma2w2 = theta(9);
    sigma2w3 = theta(10);
    pw1w2 = theta(11);
    theta12 = theta(12);
    theta13 = theta(13);

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

        %Sample from APS3A copula using Rosenblatt transforms
    %syms u1 u2 u3 theta12 theta13
    %F_c12 = @(u1, u2, theta) u1*u2 + theta*u1*u2*(1 - u1)*(1 - (4*u2^2 - 6*u2 +3)); %CDF of APS2A copula
    %f_c12 = @(u1, u2, theta) 1 + theta*(1-2*u1)*(1- 12*(u2 - 0.5)^2); %density of c12 APS2A copula
    %f_c13 = @(u1, u3, theta) 1 + theta*(1-2*u1)*(1- 12*(u3 - 0.5)^2); %density of c13 APS2A copula
    %c12_deriv_u1 = diff(F_c12, u1); %Partial derivative of APS2A copula CDF w.r.t first argument
    c12_deriv_u1 = @(u1, u2, theta) u2 + theta*u1*u2*(4*u2^2 - 6*u2 + 2) + theta*u2*(u1 - 1)*(4*u2^2 - 6*u2 + 2);
    %c13_int = int(f_c13, u3, 0, u3);
    c13_int = @(u1, u3, theta) 4*theta*(2*u1 - 1)*u3^3 - 6*theta*(2*u1 - 1)*u3^2 + (2*theta*(2*u1 - 1) + 1)*u3;
    gauss_copula = @(pw1w2, u2, u3) copulapdf('Gaussian',[u2 u3], [1,pw1w2;pw1w2,1]);
    APS3A_copula_samples = zeros(S_kernel, 3);
    for h=1:S_kernel
        rng(h)
        PPP = rand(1,3);
        z1 = PPP(1);

        c12_deriv_u1_inverse_func = @(u2) c12_deriv_u1(z1, u2, theta12) - PPP(2);
        z2 = fzero(c12_deriv_u1_inverse_func, [0 1]);

        %c13_int_func = @(u3) f_c13(x1, u3, theta13);
        %int_c13 = @(u3) integral(c13_int_func, 0, u3, 'ArrayValued',1);
        gauss_int_func = @(u3) gauss_copula(pw1w2, z2, u3);
        int_c23 = @(u3) integral(gauss_int_func, 0, u3, 'ArrayValued',1);
        x3_inverse_function = @(u3) (c13_int(z1, u3, theta13) + int_c23(u3) - u3) - PPP(3);
        %x3_inverse_function = @(u3) (u3 + f_c12(x1, x2, theta12)*u3 - u3 + int_c13(u3) - u3 + int_c23(u3) - u3)/f_c12(x1, x2, theta12) - PPP(3);
        z3 = fzero(x3_inverse_function, [0 1]);

        APS3A_copula_samples(h,:) = [z1, z2, z3];
    end

    rng(10) %Set seed
    simulated_v = normrnd(0, sqrt(sigma2v), S_kernel, 1);
    simulated_u = sqrt(sigma2u)*norminv((APS3A_copula_samples(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_w2 = norminv(APS3A_copula_samples(:,2), mu2, sqrt(sigma2w2));
    simulated_w3 = norminv(APS3A_copula_samples(:,3), mu3, sqrt(sigma2w3));
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
    
        %V[u|eps, w1, w2] = E[u^2|w2, w3, eps] - (E[u|w2, w3, eps])^2
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
    
    function logL = AppLoglikelihood_APS3A(coefs)
        %tranform parameters back true range
        coefs(7:10) = exp(coefs(7:10));
        logDen = AppLogDen_APS3A(coefs);
        logL = -sum(logDen);
    end

    function logDen = AppLogDen_APS3A(coefs) 
        N = length(y); %n obs
        [S,P] = size(us); %number of simulated draws for evaluation of the integral
        
        alpha = coefs(1);
        beta1 = coefs(2);
        beta2 = coefs(3);
        beta3 = coefs(4);
        mu2 = coefs(5);
        mu3 = coefs(6);
        sigma2u = coefs(7);
        sigma2v = coefs(8);
        sigma2w2 = coefs(9);
        sigma2w3 = coefs(10);
        pw1w2 = coefs(11);
        theta12 = coefs(12);
        theta13 = coefs(13);
        
        %Check if the delta bound is violated
        %delta_bound = (1 - pw1w2^2)^(-0.5)*exp(-0.5*(pw1w2^2/(1-pw1w2^2)));
        %if ((abs(theta12) + abs(theta13) + 1e-5) > delta_bound/2) && check_theta_bounds == 1 %Add a tolerance to ensure standard errors can be estimated using OPG
            %logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if we violate the delta bound.
        %else
            %Construct sample information
        eps = y - alpha - x1*beta1 - x2*beta2 - x3*beta3;
        B2 = p2 - p1 +log(beta1) - log(beta2);
        B3 = p3 - p1 +log(beta1) - log(beta3);
        w2 = x1 - x2 - B2;
        w3 = x1 - x3 - B3; 
        CdfUs = 2*(normcdf(sqrt(sigma2u)*us, 0, sqrt(sigma2u)) -0.5);
        eps_SxN = reshape(repelem(eps, S), S, N);
        us_SxN = sqrt(sigma2u)*us;
        EpsPlusUs = eps_SxN + us_SxN;
        DenEpsPlusUs = normpdf(EpsPlusUs, 0, sqrt(sigma2v)); %Standard normal density of eps + u (random noise terms)

        RhoWW =[1,pw1w2;pw1w2,1]; %correlation matrix allocative inefficiency data for c_{23}(F(w1), F(w2)) (bivariate gaussian copula)
        %Check if Rho is positive definite
        [R,flag] = chol(RhoWW);
        if flag ~= 0
            logDen =  ones(N, 1)*-inf; %Assign an arbitrarily large log density if Sigma is not positve definite
        else
            DenW2 = normpdf(w2, mu2, sqrt(sigma2w2));
            DenW3 = normpdf(w3, mu3, sqrt(sigma2w3));
            %Evaluate the integral via simulation (to integrate out u from eps+u)
            simulated_copula_pdfs = zeros(S,100);
            w2_rep = reshape(repelem(w2, S),S, N);
            w3_rep = reshape(repelem(w3, S),S, N);
                %Compute the CDF (standard normal) for repeated allocative inefficiency terms
            CDF_w2_rep = normcdf(w2_rep, mu2, sqrt(sigma2w2));
            CDF_w3_rep = normcdf(w3_rep, mu3, sqrt(sigma2w3));       
            for j = 1:N
                simulated_c12_pdf = 1 + theta12.*(1-2.*CdfUs(:,j)).*(1 - 12.*(CDF_w2_rep(:,j) - 0.5).^2);
                simulated_c13_pdf = 1 + theta13.*(1-2.*CdfUs(:,j)).*(1 - 12.*(CDF_w3_rep(:,j) - 0.5).^2);
                simulated_c23_pdf = copulapdf('Gaussian',[CDF_w2_rep(:,j),CDF_w3_rep(:,j)], RhoWW);
                APS3A_copula_pdf = 1 + (simulated_c12_pdf -1) + (simulated_c13_pdf - 1) + (simulated_c23_pdf - 1);
                simulated_copula_pdfs(:,j) = APS3A_copula_pdf;
            end
            Integral = mean(simulated_copula_pdfs.*DenEpsPlusUs)'; %Evaluation of the integral over S simulated samples. Column-wise mean.
            %Remove this later
            %Integral(Integral < 0) = 1e-6;
            DenAll = DenW2.*DenW3.*Integral; %Joint desnity. product of marginal density of w2 and w3 and the joint density f(\epsilon, w)
            %DenAll(abs(DenAll) < 1e-6) = 1e-6; %Adjust any exceptionally small densities to be approximatly zero

            r = beta1 + beta2 + beta3;    
            logDen = log(r)+log(DenAll);
        end
    end
end