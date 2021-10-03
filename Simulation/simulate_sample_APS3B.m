function APS3B_copula_samples = simulate_sample_APS3B(theta12, theta13, pw1w2, n, random_seed)
    gauss_copula = @(pw1w2, u2, u3) copulapdf('Gaussian',[u2 u3], [1,pw1w2;pw1w2,1]);
    APS3B_copula_samples = zeros(n,3);
 
    for h=1:n
        rng(random_seed*h)
        PPP = rand(1,3);
        x1 = PPP(1);

        c12_deriv_u1_inverse_func = @(u2) APSA3B_conditional_u1(x1, u2, theta12) - PPP(2);
        x2 = fzero(c12_deriv_u1_inverse_func, [0 1]);

        gauss_int_func = @(u3) gauss_copula(pw1w2, x2, u3);
        int_c23 = @(u3) integral(gauss_int_func, 0, u3, 'ArrayValued',1);
        x3_inverse_function = @(u3) (APSA3B_integral_u3(x1, u3, theta13) + int_c23(u3) - u3) - PPP(3);
        x3 = fzero(x3_inverse_function, [0 1]);

        APS3B_copula_samples(h,:) = [x1, x2, x3];
    end
end