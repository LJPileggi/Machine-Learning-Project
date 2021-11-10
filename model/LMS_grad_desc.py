#pylint: disable = C0114, C0103, R0913

from math import floor
import random as rand
import numpy as np
from nn_unit import nn_unit


def _LMS_step(unit, x, y):
    """
    Implements gradient descent algorithm on lms error.
    Updates weight vectors until convergence is reached
    or lms error goes below a fixed threshold (thr).

    params:
     - x: input vector; class type: numpy.ndarray;
     - y: output vector; class type: float;
      class type: float;

     returns:
      - delta_w; class type: numpy.ndarray.
    """
    error_signal = (y - unit.out(x))
    return error_signal*unit.out_prime(x)*x

def _mean_square_error(TS, unit):
        errors = [(unit.out(x) - y)**2 for (x, y) in TS]
        mse = sum(errors)/len(errors)
        #print("mse", mse)
        return mse

def grad_descent_LMS(TS, unit, eta=0.01, thr=0.001, N_max=10000, batch=False):
    """
    Implements gradient descent algorithm on lms error.
    Updates weight vectors until convergence is reached
    or lms error goes below a fixed threshold (thr).

    params:
     - TS: the training set, consisting in a list of couples (pattern, label)
     - eta: learning rate, must lie between 0 and 1; default value: 0.5;
     class type: float;
     - thr: learning threshold, if Err < thr then returns; must be
     positive; default value: 0.01; class type: float;
     - N_max: control parameter, sets the maximum number of iterations
     handlable; must be positive integer; default value: 1000;
     class type: int.

     returns:
      - updated weight vectors; class type: numpy.ndarray.
    """
    epochs = 1
    while epochs <= N_max and _mean_square_error(TS, unit) > thr:
       #print("epoch ", epochs)
       if batch == False:
            for x, y in TS:
                delta_w = _LMS_step(unit, x, y)
                unit.update(eta*delta_w)
                epochs += 1
       else:
            total_delta_w = np.zeros(3)
            for x, y in TS:
                delta_w = _LMS_step(unit, x, y)
                total_delta_w = total_delta_w + delta_w
            unit.update(eta*(1/50)*total_delta_w)
            epochs += 1



def main():
    TS = []
    b = 5
    for i in range(50):
        x = rand.random()
        y = rand.random()
        out = 3*x + y +10 + 0.1*rand.gauss(0., 1)
        TS.append((np.array([1., x, y]), out))
    unit = nn_unit(3)
    grad_descent_LMS(TS, unit)
    print(unit.w)

if __name__ == "__main__":
    main()


 
    # w_old = w_init
    # i = floor(N_max)
    # while i > 0:
    #     if Error(x, w_old, y) < thr:
    #         return w_old
    #     w_new = w_old - eta*D_Error(x, w_old, y)
    #     w_old = w_new
    #     i -= 1
    # return w_old


# def Error(x, w, y):
    
#     Empirical error for least mean squares regression.

#     params:
#      - x: input vector; class type: numpy.ndarray;
#      - w: weight vector; class type: numpy.ndarray;
#      - y: output vector; class type: float.
    
#     #error handling
#     if not isinstance(type(x), np.ndarray):
#         raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
#     if not isinstance(type(w), np.ndarray):
#         raise TypeError(f'TypeError: argument w must be <{np.ndarray}>, not <{type(w)}>')
#     if not isinstance(type(y), float):
#         raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
#     ###################
#     err = (y - (x*w).sum())**2
#     return err

# def D_Error(x, w, y):
    
#     Gradient of mean square error.

#     params:
#      - x: input vector; class type: numpy.ndarray;
#      - w: weight vector; class type: numpy.ndarray;
#      - y: output vector; class type: float.
    
#     #error handling
#     if not isinstance(type(x), np.ndarray):
#         raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
#     if not isinstance(type(w), np.ndarray):
#         raise TypeError(f'TypeError: argument w must be <{np.ndarray}>, not <{type(w)}>')
#     if not isinstance(type(y), float):
#         raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
#     ###################
#     D_err = ((x*w).sum() - y)*x
#     return D_err



# TO BE COMPLETED
# def LMS_grad_desc_batch(x, w_init, y, eta=0.5, thr=0.01, N_max=1000):
    
#     Batch version of gradient descent algorithm. At each step,
#     gradient runs over all l given patterns.

#     params:
#      - x: input vector; class type: numpy.ndarray of shape (l, n);
#      - w_init: initialised weight vector; class type: numpy.ndarray of shape (n);
#      - y: output vector; class type: numpy.ndarray of shape (l);
#      - eta: learning rate, must lie between 0 and 1; default value: 0.5;
#      class type: float;
#      - thr: learning threshold, if Err < thr then returns; must be
#      positive; default value: 0.01; class type: float;
#      - N_max: control parameter, sets the maximum number of iterations
#      handlable; must be positive integer; default value: 1000;
#      class type: int.

#      returns:
#       - updated weight vectors; class type: numpy.ndarray.
    
#     #error handling
#     if not isinstance(type(x), np.ndarray):
#         raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
#     if not isinstance(type(w_init), np.ndarray):
#         raise TypeError(f'TypeError: argument w_init must be <{np.ndarray}>, not <{type(w_init)}>')
#     if not isinstance(type(y), float):
#         raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
#     if not (eta >= 0.) | (eta <= 1.):
#         raise ValueError('ValueError: eta must fall between 0. and 1.')
#     if thr < 0.:
#         raise ValueError('ValueError: thr must be positive')
#     if N_max < 0:
#         raise ValueError('ValueError: N_max must be positive integer')
#     if not x.shape[0] == y.shape[0]:
#         raise ValueError('ValueError: x.shape[0] must equal y.shape[0]
#     if not x.shape[1] == w_init.shape[0]:
#         raise ValueError('ValueError: x.shape[1] must equal w_init.shape[0]
#     w_old = w_init
#     i = floor(N_max)
#     while i > 0:
#         #create loop
#         if Error(x, w_old, y) < thr:
#             return w_old
#         w_new = w_old - eta*D_Error(x, w_old, y)
#         w_old = w_new
#         i -= 1
#     return w_old
#     """


            
            
        
        
