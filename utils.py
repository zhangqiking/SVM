import numpy as np 

def kernel(x_i, x_j, kernel_type='rbf'):
    """
    Calculate kernel  value

    Parameters
    ----------
    x_i: array_like
         The shepe of x_i (feature_dimension, 1)
    x_j: array_like
         The shepe of x_j (feature_dimension, 1)
    kernel_type: string_like
         give the kernel function type
    
    Returns
    ----------
    kernel_value: double_like
    """
    if kernel_type == None:
        kernel_value = np.dot(x_i, x_j)
    elif kernel_type == 'poly':
        kernel_value = (np.dot(x_i, x_j) + 1)**0.2
    elif kernel_type == 'rbf':
        kernel_value = np.exp((-1 * np.linalg.norm(x_i - x_j)**2) / 0.25)
    else:
        pass
    
    return kernel_value

def range_L_H(C, alpha_1, alpha_2, y_1, y_2):
    """
    Calculate the range of alpha_2_new limited by L and H

    Parameters
    ----------
    C:  float_like
        Penalty parameters
    alpha_1: float_like
        target parameter one 
    alpha_2: float_like
        target paramater two
    y_1: float_like
        train label one
    y_2: float_like
        train_label two
    
    Returns
    ---------
    L: float_like
       low bound 
    H: float_like
       high bound
    """
    if y_1 != y_2:
        L = max(0, alpha_2 - alpha_1)
        H = min(C, C + alpha_2 - alpha_1)
    else:
        L = max(0, alpha_2 + alpha_1 - C)
        H = min(C, alpha_2 + alpha_1)
    
    return (L, H)
    




