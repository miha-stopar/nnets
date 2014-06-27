import numpy as np
from neuralnet import NN

class Recurrent(NN):
    """
    DON'T FORGET TO SET "shuffle=False" when training recurrent neural networks!
    """
    
    def __init__(self, units_per_layer, activation_functions, cost_function, language_model=False):
        # the first hidden layer will be concatenated with the input layer
        u_per_layer = units_per_layer[:]
        u_per_layer[0] += units_per_layer[1]
        self.units_per_layer = u_per_layer
        self.cache = None
        self.language_model = language_model
        super(Recurrent, self).__init__(u_per_layer, activation_functions, cost_function)
        
    def train(self, uinputs, utargets, batch_size, alpha=1, lamda=0.0, iterations = 1000, calculate_errors=False):
        # shuffle=False !
        epoch_errors = super(Recurrent, self).train(uinputs, utargets, batch_size, alpha, lamda, iterations, shuffle=False, 
                                                    calculate_errors=calculate_errors)
        return epoch_errors

    def feed_forward(self, weights, batch_inputs):
        extension = None
        if self.cache != None:
            extension = self.cache
        else:
            #extension = np.zeros(self.units_per_layer[1]) # for one input at a time - probably this is all that is needed anyway
            extension = np.zeros((len(batch_inputs), self.units_per_layer[1]))
            extension += 0.1
        if extension.ndim == 1:
            extension = [extension]
        extended_batch_inputs = np.hstack((batch_inputs, extension))
        out = super(Recurrent, self).feed_forward(weights, extended_batch_inputs)
        #self.cache = self.a[2][0][1:] # remove bias
        self.cache = self.a[2][:, 1:] # remove bias
        return out
    
    
    
    
    