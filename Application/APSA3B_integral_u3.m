function density = APSA3B_integral_u3(u1, u3, theta)
    if u3 <= 0.5
        density = u3*(2*theta*u1 - theta + 2*theta*u3 - 4*theta*u1*u3 + 1);
    else
        density = u3 - theta + 2*theta*u1 + 3*theta*u3 - 2*theta*u3^2 + 4*theta*u1*u3^2 - 6*theta*u1*u3;
    end
end