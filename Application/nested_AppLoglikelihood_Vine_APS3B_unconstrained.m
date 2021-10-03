function [logMLE, theta, sterr, u_hat, V_u_hat, non_par_u_hat, u_hat_conditional_w2w3, V_u_hat_conditional_w2w3, V_non_par_u_hat] = nested_AppLoglikelihood_Vine_APS3B_unconstrained(y, x1, x2, x3, p1, p2, p3, us, theta0)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE] = fminunc(@AppLoglikelihood_Vine_APS3B, theta0, Options); %Maximise the log-likelihood
%     rng(10) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display','off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_Vine_APS3B,'options',Options);
%     [theta, logMLE] = run(gs, problem);  
    
    %transform parameters back to true range
    theta(7:10) = exp(theta(7:10));
%     b = 1;
%     a = -1;
%     theta(12:13) = ((b-a)/2).*tanh(theta(12:13)) + (b+a)/2;
%     theta(12:13) =((b-a)/2)*((2/pi).*atan(theta(12:13))) + (b+a)/2;
    
    %Estimation of standard errors 
    delta = 1e-6;
    grad = zeros(length(y), length(theta));
    for i=1:length(theta)
        theta1 = theta;
        theta1(i) = theta(i) + delta;
        grad(:,i) = (AppLogDen_Vine_APS3B(theta1, 0) - AppLogDen_Vine_APS3B(theta, 0))/delta; 
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
    
    %Sample from the APS2B Vine Copula using conditional sampling
    APS2B_vine_copula_samples = zeros(S_kernel,3);
            %Partial derivative of Gaussian Copula w.r.t second argument
    C12_condtional_u2 = @(u1_, u2_, pw1w2_) normcdf((norminv(u1_) - pw1w2_*norminv(u2_))/(sqrt(1 - pw1w2_^2)));
            %Partial derivative of APS2A copula w.r.t second argument (conditional cdf)
%     syms u1 u2 theta
%     APS2B_CDF1 = u1*u2 + theta*u1*(1 - u1)*(u2 - 2*u2 + 2*u2^2);
%     APS2B_CDF2 = u1*u2 + theta*u1*(1 - u1)*(u2 - 1 + 2*u2 - 2*u2^2);
%     APS2B_CDF = piecewise(u2 <= 0.5, APS2B_CDF1, u2 > 0.5, APS2B_CDF2);
%     PD = diff(APS2B_CDF, u2);
    copula_PDs_matrix = {C12_condtional_u2 @APS2B_conditional_u2; @APS2B_conditional_u2 0};
    copula_theta_matrix = [pw1w2 theta12; theta13 0];
    n_ = 3;
    for h = 1:S_kernel
        rng(h)
        PPP = rand(1,3);
        X_vector = zeros(n_,1);
        v_matrix = [];
        v_matrix(1,1) = PPP(1);
        X_vector(1) = PPP(1);
        for i = 2:3
            v_matrix(i,1) = PPP(i);
            for k = i-1:-1:1
                inverse_function = @(u1_) copula_PDs_matrix{k,i-k}(u1_, v_matrix(k,k), copula_theta_matrix(k,i-k)) - v_matrix(i,1);
                v_matrix(i,1) = fzero(inverse_function, [0 1]);
            end
            X_vector(i) = v_matrix(i,1);
            if i == n_
                break
            end
            for j = 1:i-1
                v_matrix(i, j+1) = copula_PDs_matrix{j, i-j}(v_matrix(i,j), v_matrix(j,j), copula_theta_matrix(j, i-j));
            end
        end
        APS2B_vine_copula_samples(h,:) = [X_vector(3), X_vector(1), X_vector(2)];
    end
    
    rng(10) %Set seed
    simulated_v = normrnd(0, sqrt(sigma2v), S_kernel, 1);
    simulated_u = sqrt(sigma2u)*norminv((APS2B_vine_copula_samples(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_w2 = norminv(APS2B_vine_copula_samples(:,2), mu2, sqrt(sigma2w2));
    simulated_w3 = norminv(APS2B_vine_copula_samples(:,3), mu3, sqrt(sigma2w3));
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)  

           %Bandwidth information for each conditioning variable
    h = 1.06*n^(-1/5)*[max(std(obs_eps), iqr(obs_eps)/1.34) max(std(obs_w2), iqr(obs_w2)/1.34) max(std(obs_w3), iqr(obs_w3)/1.34)];
 
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
    
            %kernel variance estimation
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
    
    function logL = AppLoglikelihood_Vine_APS3B(coefs)
        %tranform parameters back true range
        coefs(7:10) = exp(coefs(7:10));
%         b = 1;
%         a = -1;
%         coefs(12:13) = ((b-a)/2).*tanh(coefs(12:13)) + (b+a)/2;
%         coefs(12:13) =((b-a)/2)*((2/pi).*atan(coefs(12:13))) + (b+a)/2;
        logDen = AppLogDen_Vine_APS3B(coefs, 1);
        logL = -sum(logDen);
    end

    function logDen = AppLogDen_Vine_APS3B(coefs, check_theta_bounds) 
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
        
        %Check if the bounds on theta12 and theta13 are violated
        %if ((theta12 - 1e-5 < -1 || theta12 + 1e-5 > 1) || (theta13 - 1e-5 < -1 || theta13 + 1e-5 > 1)) && check_theta_bounds == 1 %Add a tolerance to ensure standard errors can be estimated using OPG
            %logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if we violate the bound on theta12/theta13.
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

        DenW2 = normpdf(w2, mu2, sqrt(sigma2w2)); %marginal density of allocative inefficiency terms
        DenW3 = normpdf(w3, mu3, sqrt(sigma2w3)); %marginal density of allocative inefficiency terms 
        RhoWW =[1,pw1w2;pw1w2,1]; %correlation matrix allocative inefficiency data for c_{23}(F(w1), F(w2)) (bivariate gaussian copula)
        %Check if Rho is positive definite
        [R,flag] = chol(RhoWW);
        if flag ~= 0
            logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if Sigma is not positve definite
        else
            %Evaluate the integral via simulation (to integrate out u from eps+u)
            simulated_copula_pdfs = zeros(S,100);
            w2_rep = reshape(repelem(w2, S),S, N);
            w3_rep = reshape(repelem(w3, S),S, N);
            %Compute the CDF (standard normal) for repeated allocative inefficiency terms
            CDF_w2_rep = normcdf(w2_rep, mu2, sqrt(sigma2w2));
            CDF_w3_rep = normcdf(w3_rep, mu3, sqrt(sigma2w3));     
             %Compute the conditional CDFs for allocative inefficiency terms F(w1|u), F(w2|u)
            conditional_CDF_w2_u = zeros(S, N);
            conditional_CDF_w3_u = zeros(S, N);
            for j = 1:N
               if w2(j) <= 0.5
                    conditional_CDF_w2_u(:,j) = CDF_w2_rep(:,j) + theta12.*CdfUs(:,j).*(-2.*CDF_w2_rep(:,j).^2 + CDF_w2_rep(:,j)) + theta12.*(-2.*CDF_w2_rep(:,j).^2 + CDF_w2_rep(:,j)).*(CdfUs(:,j) - 1);
                else
                    conditional_CDF_w2_u(:,j) = CDF_w2_rep(:,j) + theta12.*(CdfUs(:,j) - 1).*(2.*CDF_w2_rep(:,j).^2 - 3.*CDF_w2_rep(:,j) + 1) + theta12.*CdfUs(:,j).*(2.*CDF_w2_rep(:,j).^2 - 3.*CDF_w2_rep(:,j) + 1);
               end
                if w3(j) <= 0.5
                    conditional_CDF_w3_u(:,j) = CDF_w3_rep(:,j) + theta13.*CdfUs(:,j).*(-2.*CDF_w3_rep(:,j).^2 + CDF_w3_rep(:,j)) + theta13.*(-2.*CDF_w3_rep(:,j).^2 + CDF_w3_rep(:,j)).*(CdfUs(:,j) - 1);
                else
                    conditional_CDF_w3_u(:,j) = CDF_w3_rep(:,j) + theta13.*(CdfUs(:,j) - 1).*(2.*CDF_w3_rep(:,j).^2 - 3.*CDF_w3_rep(:,j) + 1) + theta13.*CdfUs(:,j).*(2.*CDF_w3_rep(:,j).^2 - 3.*CDF_w3_rep(:,j) + 1);
                end
            end
            for j = 1:N
                simulated_c21_pdf = 1 + theta12.*(1-2.*CdfUs(:,j)).*(1 - 4.*abs(CDF_w2_rep(:,j) - 0.5));
                simulated_c31_pdf = 1 + theta13.*(1-2.*CdfUs(:,j)).*(1-4.*abs(CDF_w3_rep(:,j) - 0.5));
                simulated_c23_pdf = copulapdf('Gaussian',[conditional_CDF_w2_u(:,j),conditional_CDF_w3_u(:,j)], RhoWW);
                APS3B_copula_pdf = simulated_c21_pdf.*simulated_c31_pdf.*simulated_c23_pdf;
                simulated_copula_pdfs(:,j) = APS3B_copula_pdf;
            end
            Integral = mean(simulated_copula_pdfs.*DenEpsPlusUs)'; %Evaluation of the integral over S simulated samples. Column-wise mean.
            DenAll = DenW2.*DenW3.*Integral; %joint desnity. product of marginal density of w_{1} and w_{2} and the joint density f(\epsilon, w)
            %DenAll(abs(DenAll) < 1e-6) = 1e-6; %Adjust any exceptionally small densities to be approximatly zero for logorithm

            r = beta1 + beta2 + beta3;
            logDen = log(r) + log(DenAll);
        end 
    end 
end