function [theta, sterr, u_hat, u_hat_conditional_w1w2] = nested_AppLoglikelihood_QMLE(y, x1, x2, z1, z2, theta0)
    Options = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'Off');
    [theta, logMLE, ECMLE, output, grad, hessian] = fminunc(@AppLoglikelihood_QMLE, theta0, Options);
%      rng(10) %Random seed for global search problem
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%     problem = createOptimProblem('fmincon','x0',theta0, 'objective',@AppLoglikelihood_QMLE,'options',Options, ...
%                                  'lb',[0, 0, 0, log(0.5), log(0.5), log(0.5), log(0.5), 0.5], ...
%                                  'ub',[1, 1, 1, log(1.5), log(1.5), log(1.5), log(1.5), 1.5]);
%     [theta, logMLE] = run(gs, problem);
        
    %Estimate standard errors
    theta(4:7) = exp(theta(4:7)); %Transform parameters back to the true range
        %Compute contribution of each observation to the gradient of each parameter
    delta = 1e-6;
    grad_contributions = zeros(length(y), length(theta));
    for i=1:length(theta)
        theta1 = theta;
        theta1(i) = theta(i) + delta;
        grad_contributions(:,i) = (AppLogDen_QMLE(theta1)-AppLogDen_QMLE(theta))/delta; 
    end
    OPG = grad_contributions'*grad_contributions; 
%     robust_covariance_matrix = inv(hessian)*OPG*inv(hessian);
%     sterr = sqrt(diag(robust_covariance_matrix));
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
    u_hat_conditional_w1w2 = 0;
    
    function logL = AppLoglikelihood_QMLE(coefs)
        % tranform parameters back true range
        coefs(4:7) =  exp(coefs(4:7));
        logDen = AppLogDen_QMLE(coefs);
        logL = -sum(logDen);
    end

    function logDen = AppLogDen_QMLE(coefs)
        N = length(y);
        % obtain the log likelihood
        alpha = coefs(1);
        beta1 = coefs(2);
        beta2 = coefs(3);
        sigma2v = coefs(4); 
        sigma2u = coefs(5);
        sigma2w1 = coefs(6);
        sigma2w2 = coefs(7);
        gamma = coefs(8);

        lambda = sqrt(sigma2u/sigma2v); %standard error of the ratio of variance of technical and random errors.
        sigma2 = sigma2u+sigma2v; %variance of the composed error term e = v + u
        sigma = sqrt(sigma2);

        eps = y - alpha - x1.*beta1 - x2.*beta2; %composed errors from the production function equation (i.e residuals from the production function)
        w1 = x1 - z1*gamma; %re-arranged endogenous regressor equations p.11
        w2 = x2 - z2*gamma; %re-arranged endogenous regressor equations p.11
        w = [w1,w2];
        %Compute densities
        Den = 2/sigma*normpdf(eps/sigma, 0, 1).*(1 - normcdf(lambda*eps/sigma, 0, 1)); %density of composed error (SF model residuals) eq. (8) ALS77
        logDen_Tech = log(Den); %log density of composed error (SF model residuals)
%         SigmaWW = w'*w/length(y); %covariance matrix for allocative inefficiency terms.
%         SigmaWW(1,1) = sqrt(sigma2w1);
%         SigmaWW(2,2) = sqrt(sigma2w2);
        pw1w2 = corr(w1,w2);
        SigmaWW = [sqrt(sigma2w1), pw1w2*sqrt(sigma2w1)*sqrt(sigma2w2); pw1w2*sqrt(sigma2w1)*sqrt(sigma2w2), sqrt(sigma2w2)];
        %Check if Sigma is positive definite
        [R,flag] = chol(SigmaWW);
        if flag ~= 0
            logDen = ones(N, 1)*-inf; %Assign an arbitrarily large log density if Sigma is not positve definite
        else
            logDen_Alloc = log(mvnpdf(w, 0, SigmaWW)); %log multivariate normal density. by assumption the allocative inefficieny vector is multivariate normal.
            logDen = logDen_Tech + logDen_Alloc;
        end
    end 
end