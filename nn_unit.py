#pylint: disable = C0114, C0103

import numpy as np

class nn_unit:
    """Object class for the NN unit
    params:
     - x: input vector; class type: numpy.ndarray;
     - w_i: weight vector; class type: numpy.ndarray;
     - f_i: activation function; class type: function.

    properties:
     - net_i: scalar product between x and w_i; class type: float;
     - o_i: output vector; class type: float or numpy.ndarray.
    """
    typecode = 'f'
    def __init__(self, x, w_i, f_i):
        if not isinstance(type(x), np.ndarray):
            raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
        else:
            self.x = x
        if not isinstance(type(w_i), np.ndarray):
            raise TypeError(f'TypeError: argument w_i must be <{np.ndarray}>, not <{type(w_i)}>')
        else:
            self.w_i = w_i
        self.f_i = f_i

    @property
    def net_i(self):
        """net_i
        returns argument for activation function f. Type: float
        """
        return float((self.x * self.w_i).sum())

    @property
    def o_i(self):
        """o_i
        returns output units. Type: either float or np.ndarray
        """
        return self.f_i(self.net_i)
