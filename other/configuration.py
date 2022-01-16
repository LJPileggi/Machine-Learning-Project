# import copy
# import itertools

# class Configuration:

#     def __init__(self, layers, batch_size, eta_decay, eta, lam, alpha):
#           self.layers       = layers
#           self.batch_size   = batch_size 
#           self.eta_decay    = eta_decay
#           self.eta          = eta
#           self.lam          = lam 
#           self.alpha        = alpha
    
#     def __repr__(self):
#        return f'layers: {self.layers}, batch: {self.batch_size}, eta {self.eta}, eta_decay {self.eta_decay}, lambda {self.lam}, alpha  {self.alpha}'    

#     def get_copy_with(self, hyper_dict):
#         out = copy.deepcopy(self)
#         out.layers = hyper_dict["layers"]
#         out.batch_size = hyper_dict["batch_size"]
#         out.eta_decay = hyper_dict["eta_decay"]
#         out.eta = hyper_dict["eta"]
#         out.lam = hyper_dict["lam"]
#         out.alpha = hyper_dict["alpha"]
#         return out
        
#     def my_product(inp):
#         return [dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values())]
