import numpy as np


def expect(p):
    n= 20
    if p==1:
        return (n-1)/2
    else:
        return (p-p**n)/(1-p)/(1-p**n) - p**n*(n-1)/(1-p**n)
    
def get_param(E):
    from scipy.optimize import fsolve
    def solve_func(x, f, target_y):
        return f(x) - target_y

    x0 = 0 
    x_solution = fsolve(solve_func, x0, args=(expect, E))
    # print(x_solution)
    # print(expect(x_solution))
    return x_solution
def prob(p, n):
    prod = 1
    # p = np.exp(-t)
    
    sample_ratio = []
    
    for j in range(n):
        sample_ratio.append(prod)
        prod *= p
    
    return sample_ratio
    
    # return prod
    
def distribution(p):
    # Calculate the sum of alpha_i(t) for all i
    n = 20
    
    weights = prob(p, n)
    # print(weights)
    weights = np.array(weights)
    # print(np.sum(weights))
    weights = weights / np.sum(weights)

    # Return the sum divided by M
    return weights