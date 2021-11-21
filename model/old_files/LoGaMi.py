



class LoGaMi ():
    """
    Constructor of out NN model

    params:
     - model_config: dictionary with the model configuration, already parsed from the json under configs that was choosen
     - device: the cuda device we decided to use for this NN
    """
    def __init__ (self, model_config, device):
        self.device = device

        self.lambda = model_config["lambda"]
        self.epsilon = model_config["epsilon"]
        self.eta = model_config["eta"]

        #na cosa del genere

    """
    Funzione di Forward

    """
    def forward ():

    """
    Funzione di Backward

    """
    def backward ():
