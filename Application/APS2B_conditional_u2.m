function CDF = APS2B_conditional_u2(u1, u2, theta)
    if u2 <= 0.5
        CDF =  u1 - theta*u1*(4*u2 - 1)*(u1 - 1);
    elseif u2 > 0.5
        CDF = u1 + theta*u1*(4*u2 - 3)*(u1 - 1);
    end
end
