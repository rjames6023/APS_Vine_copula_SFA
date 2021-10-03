function Sigma = CholeskiToMatrix(CholOfSigma) 
    SigmaChol = zeros(3,3); %Construct empty covariance matrix
    SigmaChol(itril(size(SigmaChol))) = CholOfSigma; %extract and insert lower triangular cholesky decomposition
    Sigma = SigmaChol*SigmaChol'; 
end