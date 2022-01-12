import copy

class Configuration:
    k = 10000

    def __init__(self, layers, batch_size, eta_decay, eta, lam, alpha):
          self.layers       = layers
          self.batch_size   = batch_size 
          self.eta_decay    = eta_decay
          self.eta          = eta
          self.lam          = lam 
          self.alpha        = alpha
    
    def get_copy_with(self, eta, lam, alpha):
        out = copy.deepcopy(self)
        out.eta = eta
        out.lam = lam
        out.alpha = alpha
        
    def my_product(inp):
        return (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))