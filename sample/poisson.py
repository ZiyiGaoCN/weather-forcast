import numpy as np



def posson_i(t, i):
    prod = 1
    for j in range(0, i):
        prod = prod * t / (j+1) 
    return prod
    
def distribution(t):
    # Calculate the sum of alpha_i(t) for all i
    n = 20
    weights = []
    for i in range(0, n):
        weights.append(posson_i(t, i))
    weights = np.array(weights)
    print(np.sum(weights))
    weights = weights / np.sum(weights)

    # Return the sum divided by M
    return weights