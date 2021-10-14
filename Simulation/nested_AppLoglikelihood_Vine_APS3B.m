function [theta, sterr, u_hat, non_par_u_hat, u_hat_conditional_w1w2] = nested_AppLoglikelihood_Vine_APS3B(y, x1, x2, z1, z2, us, theta0, random_seed)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE, grad, hessian] = fminunc(@AppLoglikelihood_Vine_APS3B, theta0, Options);
%     rng(random_seed) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_Vine_APS3B,'options',Options, ...
%                                  'lb',[0, 0, 0, log(0.1), log(0.1), log(0.1), log(0.1), 0.1, -0.99, -0.99, -0.99], ...
%                                  'ub',[1, 1, 1, log(1.5), log(1.5), log(1.5), log(1.5), 1.5, 0.99, 0.99, 0.99]);
%     [theta, logMLE] = run(gs, problem);
    
    %Transform parameters back to the true range
    theta(4:7) = exp(theta(4:7));
%     b = 1;
%     a = -1;
%     theta(10:11) = ((b-a)/2).*tanh(theta(10:11)) + (b+a)/2;
    
    %Estimate standard errors 
        %Trim the final 3 columns from the hessian (remove parameter rho and theta12 theta13)
%     [hessian_N, hessian_P] = size(hessian);
%     hessian = hessian(:,1:hessian_P-3);
    
    delta = 1e-6;
    grad_contributions = zeros(length(y), length(theta)-3);
    for i=1:length(theta)-3
        theta1 = theta;
        theta1 = [theta1', theta(end-2), theta(end-1), theta(end)]; %Add back rho and pw1w2 for density estimation
        theta1(i) = theta(i) + delta;
        grad_contributions(:,i) = (AppLogDen_Vine_APS3B(theta1, 0)-AppLogDen_Vine_APS3B(theta, 1))/delta; 
    end
    OPG = grad_contributions'*grad_contributions; 
    sterr = sqrt(diag(inv(OPG)));

    %Prediction of technical inefficiency
    alpha = theta(1);
    beta1 = theta(2);
    beta2 = theta(3);
    sigma2v = theta(4); 
    sigma2u = theta(5);
    sigma2w1 = theta(6);
    sigma2w2 = theta(7);
    gamma = theta(8);
    pw1w2 = theta(9);
    theta12 = theta(10);
    theta13 = theta(11);

    obs_eps = y - alpha - x1*beta1 - x2*beta2; %composed errors from the production function equation (i.e residuals from the production function)
    lambda = sqrt(sigma2u/sigma2v);
    sigma = sqrt(sigma2u+sigma2v);
    sig_star = sqrt(sigma2u*sigma2v/(sigma^2));
    u_hat = sig_star*(((normpdf(lambda.*obs_eps./sigma, 0, 1))./(1-normcdf(lambda.*obs_eps./sigma))) - ((lambda.*obs_eps)./sigma)); %Conditional distribution of u given eps. 
         
    %Prediction of technical inefficiency conditional on w1 and w2
     S_kernel = 10000; %number of simulated draws for evaluation of the expectation from joint distribution of (u,w21,w2)
     n = length(y);
        %Observed variables
    obs_w1 = x1 - z1*gamma;
    obs_w2 = x2 - z2*gamma;
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);
    rep_obs_w1 = reshape(repelem(obs_w1, S_kernel), S_kernel, n);
    rep_obs_w2 = reshape(repelem(obs_w2, S_kernel), S_kernel, n);
         
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
    rng(random_seed) %Set seed
    simulated_v = normrnd(0, sqrt(sigma2v), S_kernel, 1);
    simulated_u = sqrt(sigma2u)*norminv((APS2B_vine_copula_samples(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_w1 = norminv(APS2B_vine_copula_samples(:,2), 0, sqrt(sigma2w1));
    simulated_w2 = norminv(APS2B_vine_copula_samples(:,3), 0, sqrt(sigma2w2));
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)        
    
        %Bandwidth information for each conditioning variable
    h = 1.06*n^(-1/5)*[max(std(simulated_eps), iqr(simulated_eps)/1.34) max(std(simulated_w1), iqr(simulated_w1)/1.34) max(std(simulated_w2), iqr(simulated_w2)/1.34)];
  
    %Compute kernel estimates for E[u|eps]
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
    
    function logL = AppLoglikelihood_Vine_APS3B(coefs)
        % tranform parameters back true range
        coefs(4:7) = exp(coefs(4:7));
%         b = 1;
%         a = -1;
%         coefs(10:11) = ((b-a)/2).*tanh(coefs(10:11)) + (b+a)/2;
        logDen = AppLogDen_Vine_APS3B(coefs, 1);
        logL = -sum(logDen);
    end

    function logDen = AppLogDen_Vine_APS3B(coefs, check_theta_bounds) 
        % tranform parameters back true range
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
        pw1w2 = coefs(9);
        theta12 = coefs(10);
        theta13 = coefs(11);
        
        %Check if the bounds on theta12 and theta13 are violated
            %extended value extension method of constraining the fesible region.
        if ((theta12 - 1e-5 < -1 || theta12 + 1e-5 > 1) || (theta13 - 1e-5 < -1 || theta13 + 1e-5 > 1)) && check_theta_bounds == 1
            logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if we violate the bound on theta12/theta13.
        else
            eps = y-alpha-x1*beta1-x2*beta2; %composed errors from the production function equation (i.e residuals from the production function)
            w1 = x1 - z1*gamma; %re-arranged endogenous regressor equations p.11
            w2 = x2 - z2*gamma; %re-arranged endogenous regressor equations p.11

            %Construct technical inefficincy sample information
            eps_SxN = reshape(repelem(eps, S), S, N);
            us_SxN = sqrt(sigma2u)*us;
            CdfUs = 2*(normcdf(sqrt(sigma2u)*us, 0, sqrt(sigma2u)) -0.5);
            EpsPlusUs = eps_SxN + us_SxN;
            DenEpsPlusUs = normpdf(EpsPlusUs, 0, sqrt(sigma2v)); %Standard normal density of eps + u (random noise terms)

            %Construct allocative inefficincy sample information 
            DenW1 = normpdf(w1, 0, sqrt(sigma2w1)); %marginal density of allocative inefficiency terms
            DenW2 = normpdf(w2, 0, sqrt(sigma2w2)); %marginal density of allocative inefficiency terms 
            RhoWW=[1,pw1w2;pw1w2,1]; %correlation matrix allocative inefficiency data for c_{23}(F(w1), F(w2)) (bivariate gaussian copula)

            %Check if Sigma is positive definite
            [R,flag] = chol(RhoWW);
            if flag ~= 0
                logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if Sigma is not positve definite
            else
                %Evaluate the integral via simulation (to integrate out u from eps+u)
                simulated_copula_pdfs = zeros(S,100);
                w1_rep = reshape(repelem(w1, S),S, N);
                w2_rep = reshape(repelem(w2, S),S, N);
                    %Compute the CDF (standard normal) for repeated allocative inefficiency terms
                CDF_w1_rep = normcdf(w1_rep, 0, sqrt(sigma2w1));
                CDF_w2_rep = normcdf(w2_rep, 0, sqrt(sigma2w2));     
                     %Compute the conditional CDFs for allocative inefficiency terms F(w1|u), F(w2|u)
                conditional_CDF_w1_u = zeros(S, N);
                conditional_CDF_w2_u = zeros(S, N);
                for j = 1:N
                    if w1(j) <= 0.5
                        %conditional_CDF_w1_u(:,j) = CDF_w1_rep(:,j) + (theta12.*CDF_w1_rep(:,j)).*(1 - CDF_w1_rep(:,j)).*(4*CdfUs(:,j) - 1);
                        conditional_CDF_w1_u(:,j) = CDF_w1_rep(:,j) + theta12.*CdfUs(:,j).*(-2.*CDF_w1_rep(:,j).^2 + CDF_w1_rep(:,j)) + theta12.*(-2.*CDF_w1_rep(:,j).^2 + CDF_w1_rep(:,j)).*(CdfUs(:,j) - 1);
                    else
                        %conditional_CDF_w1_u(:,j) = CDF_w1_rep(:,j) + (theta12.*CDF_w1_rep(:,j)).*(1 - CDF_w1_rep(:,j)).*(3 - 4*CdfUs(:,j));
                        conditional_CDF_w1_u(:,j) = CDF_w1_rep(:,j) + theta12.*(CdfUs(:,j) - 1).*(2.*CDF_w1_rep(:,j).^2 - 3.*CDF_w1_rep(:,j) + 1) + theta12.*CdfUs(:,j).*(2.*CDF_w1_rep(:,j).^2 - 3.*CDF_w1_rep(:,j) + 1);
                    end
                    if w2(j) <= 0.5
                        %conditional_CDF_w2_u(:,j) = CDF_w2_rep(:,j) + (theta13.*CDF_w2_rep(:,j)).*(1 - CDF_w2_rep(:,j)).*(4*CdfUs(:,j) - 1);
                        conditional_CDF_w2_u(:,j) = CDF_w2_rep(:,j) + theta13.*CdfUs(:,j).*(-2.*CDF_w2_rep(:,j).^2 + CDF_w2_rep(:,j)) + theta13.*(-2.*CDF_w2_rep(:,j).^2 + CDF_w2_rep(:,j)).*(CdfUs(:,j) - 1);
                    else
                        %conditional_CDF_w2_u(:,j) = CDF_w2_rep(:,j) + (theta13.*CDF_w2_rep(:,j)).*(1 - CDF_w2_rep(:,j)).*(3 - 4*CdfUs(:,j));
                        conditional_CDF_w2_u(:,j) = CDF_w2_rep(:,j) + theta13.*(CdfUs(:,j) - 1).*(2.*CDF_w2_rep(:,j).^2 - 3.*CDF_w2_rep(:,j) + 1) + theta13.*CdfUs(:,j).*(2.*CDF_w2_rep(:,j).^2 - 3.*CDF_w2_rep(:,j) + 1);
                    end
                end
                for j = 1:N
                    simulated_c21_pdf = 1 + theta12.*(1-2.*CdfUs(:,j)).*(1 - 4.*abs(CDF_w1_rep(:,j)-0.5));
                    simulated_c31_pdf = 1 + theta13.*(1-2.*CdfUs(:,j)).*(1-4.*abs(CDF_w2_rep(:,j) - 0.5));
                    simulated_c23_pdf = copulapdf('Gaussian',[conditional_CDF_w1_u(:,j),conditional_CDF_w2_u(:,j)],RhoWW);
                    APS3B_copula_pdf = simulated_c21_pdf.*simulated_c31_pdf.*simulated_c23_pdf;
                    simulated_copula_pdfs(:,j) = APS3B_copula_pdf;
                end
                Integral = mean(simulated_copula_pdfs.*DenEpsPlusUs)'; %Evaluation of the integral over S simulated samples. Column-wise mean.
                %Integral(Integral < 0) = 1e-6;
                
                DenAll = DenW1.*DenW2.*Integral; %joint desnity. product of marginal density of w_{1} and w_{2} and the joint density f(\epsilon, w)
                %DenAll(abs(DenAll) < 1e-6) = 1e-6; %Adjust any exceptionally small densities to be approximatly zero
                logDen = log(DenAll);
            end
        end
    end
end