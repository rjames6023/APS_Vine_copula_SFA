function Vine_APS2B_copula_samples = simulate_sample_Vine_APS2B(theta12, theta13, pw1w2, n, random_seed)
    Vine_APS2B_copula_samples = zeros(n,3);
    
        %Partial derivative of Gaussian Copula w.r.t second argument
    C12_condtional_u2 = @(u1_, u2_, pw1w2_) normcdf((norminv(u1_) - pw1w2_*norminv(u2_))/(sqrt(1 - pw1w2_^2)));
    copula_PDs_matrix = {C12_condtional_u2 @APS2B_conditional_u2; @APS2B_conditional_u2 0};
    copula_theta_matrix = [pw1w2 theta13; theta12 0];
    
    for h =1:n
        rng(random_seed*h)
        PPP = rand(1,3);
        X_vector = zeros(3,1);
        v_matrix = [];
        v_matrix(1,1) = PPP(1);
        X_vector(1) = PPP(1);
        for i = 2:3
            v_matrix(i,1) = PPP(i);
            for k = i-1:-1:1
                inverse_function = @(u1_) copula_PDs_matrix{k,i-k}(u1_, v_matrix(k,k), copula_theta_matrix(k,i-k)) - v_matrix(i,1);
                v_matrix(i,1) = fzero(inverse_function, [0 1]);
            end
            X_vector(i,1) = v_matrix(i,1);
            if i == 3
                break
            end
            for j = 1:i-1
                v_matrix(i, j+1) = copula_PDs_matrix{j, i-j}(v_matrix(i,j), v_matrix(j,j), copula_theta_matrix(j, i-j));
            end
        end
        Vine_APS2B_copula_samples(h,:) = [X_vector(3,1), X_vector(1,1), X_vector(2,1)];
    end
end