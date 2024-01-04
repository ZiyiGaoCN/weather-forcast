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
    
def distribution():
    # Calculate the sum of alpha_i(t) for all i
    n = 20
    
    weights_1 = np.array([1/4])
    weights_2 = np.array([3/4/19]*19)
    weights = np.concatenate((weights_1, weights_2))

    # Return the sum divided by M
    return weights