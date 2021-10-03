function CDF = APS2A_conditional_u2(u1, u2, theta)
    CDF = u1 + theta*u1*(u1 - 1)*(4*u2^2 - 6*u2 + 2) + theta*u1*u2*(8*u2 - 6)*(u1 - 1);
end