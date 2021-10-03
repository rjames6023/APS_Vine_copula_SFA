function conditional_density = APSA3B_conditional_u1(u1, u2, theta)
    if u2 <= 0.5
        conditional_density = u2 + theta*u1*(- 2*u2^2 + u2) + theta*(- 2*u2^2 + u2)*(u1 - 1);
    else
        conditional_density = u2 + theta*(u1 - 1)*(2*u2^2 - 3*u2 + 1) + theta*u1*(2*u2^2 - 3*u2 + 1);
    end
end