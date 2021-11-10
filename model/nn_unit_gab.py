#pylint: disable = C0114, C0103, R0913

import numpy as np
import enum

class ActFun(enum.Enum):
    IDENTITY = 1
    TANH = 2

class nn_unit:
    """Object class for the NN unit
    params:
     - w_size: size of weight vector, which is the same to the size of anu input vector; class type: int
     - f_i: activation function; class type: function.

    properties:
     - net: scalar product between x and w_i; class type: float;
     - out: output vector; class type: float or numpy.ndarray.
    """
    typecode = 'f'
    def __init__(self, w_size, f = ActFun.IDENTITY):
        self.act_f = f
        self.w = np.random.randn(w_size)

    def _net(self,x:np.ndarray):
        """_net_i
        returns argument for activation function f. Type: float
        """
        return (x * self.w).sum()

    def out(self, x:np.ndarray):
        """out
        returns output units. Type: either float or np.ndarray
        """
        return self.act_f(self._net(x))

    def out_prime(self, x:np.ndarray):
        if self.act_f == ActFun.IDENTITY:
            return 1
        elif self.act_f == ActFun.IDENTITY:



    def update(self, delta_w:np.ndarray):
        self.w = self.w + delta_w
        
