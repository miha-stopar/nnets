from neuron.neuralnet import NN
from neuron.recurrent import Recurrent
import pylab as pl
import numpy as np
import time

if __name__ == "__main__":
    start_time = time.time()
    
    size = 100
    np.random.seed(0)
    inputs = np.linspace(-7, 7, 20)
    targets = np.sin(inputs) * 0.5
    inputs.resize((size, 1))
    targets.resize((size, 1))
    

    #nn = NN([1, 10, 1], ["tanh", "linear"], cost_function="sse")
    nn = Recurrent([1, 10, 1], ["tanh", "linear"], cost_function="sse")
    epoch_errors = nn.train(inputs, targets, batch_size=1, alpha=0.1, lamda=0.0, iterations=500, calculate_errors=True)
    
    pl.subplot(211)
    pl.plot(epoch_errors)
    pl.xlabel('epoch number')
    pl.ylabel('error')
    
    output = []
    for index, inp in enumerate(inputs):
        pred = nn.predict(inp)
        output.append(pred)
        
    x2 = np.linspace(-6.0,6.0,150)
    x2.resize((size, 1))
    output1 = []
    for index, inp in enumerate(x2):
        pred = nn.predict(inp)
        output1.append(pred)
    
    pl.subplot(212)
    pl.plot(inputs , targets, '.', inputs, output, 'p')
    pl.show()

    end = time.time()
    print "duration: %s" % (end - start_time)
    
    
