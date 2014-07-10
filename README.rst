About
=====

Python neural network library that provides the following network types:
 
 * feed-forward neural network
 * recurrent (Elman) neural network 

It supports the following activations functions:
 
 * linear
 * sigmoid
 * tanh
 * softmax
 
It supports the following cost functions:

 * sum of squared errors (SSE)
 * cross entropy (CE)
 
And it supports an arbitrary number of hidden layers and arbitrary batch size for gradient descent algorithm.

How to use
=====

To learn XOR function (see code for this and other examples in *test* folder):

::

    from neuron.neuralnet import NN
	
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    targets = [[0], [1], [1], [0]]
    nn = NN([2, 2, 1], ["sigmoid", "sigmoid"], cost_function="ce")
    nn.train(inputs, targets, batch_size=4, alpha=1, lamda=0.0, iterations=3000)
    preds = []
    for index, inp in enumerate(inputs):
        pred = nn.predict(inp)
        preds.append(pred)
        print "%s -> %s" % (inp, pred)
    
The output looks like:

::

	[0, 0] -> [ 0.00903832]
	[0, 1] -> [ 0.99312]
	[1, 0] -> [ 0.99327432]
	[1, 1] -> [ 0.00975482]

To learn (some nonsense) function using softmax output layer and two hidden layers:

::

    inputs = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    targets = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    nn = NN([3, 3, 5, 3], ["sigmoid", "tanh", "softmax"], cost_function="softmax_ce")
    nn.train(inputs, targets, batch_size=4, alpha=1, lamda=0.0, iterations=1000)
    preds = []
    for index, inp in enumerate(inputs):
        pred = nn.predict(inp)
        preds.append(pred)
        print "%s -> %s" % (inp, pred)
        
The output looks like:

::

	[0, 0, 1] -> [  9.99473157e-01   2.26014879e-04   3.00828353e-04]
	[1, 0, 0] -> [  2.29435307e-04   2.89636262e-04   9.99480928e-01]
	[0, 1, 0] -> [  1.68368019e-04   9.99476531e-01   3.55100544e-04]
	
Approximating the sine function using Recurrent network:

::

    from neuron.recurrent import Recurrent
    import pylab as pl
    import numpy as np
    
    size = 100
    np.random.seed(0)
    inputs = np.linspace(-7, 7, 20)
    targets = np.sin(inputs) * 0.5
    inputs.resize((size, 1))
    targets.resize((size, 1))

    nn = Recurrent([1, 10, 1], ["tanh", "linear"], cost_function="sse")
    epoch_errors = nn.train(inputs, targets, batch_size=1, alpha=0.1, lamda=0.0, iterations=500, calculate_errors=True)
    
    pl.subplot(211)
    pl.plot(epoch_errors)
    pl.xlabel('Epoch number')
    pl.ylabel('error (default SSE)')
    
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


.. image:: https://raw.github.com/miha-stopar/nnets/master/test/sine.png


How to find hyperparameters
=====

You can use *findparameters.find* function to try to find the optimal hyperparameters. For example for recognition of
handwritten digits (see *digits.py* and *digits_findparameters.py* in *test* folder):

::

    import scipy.io
    from neuron import findparameters

    training_data = scipy.io.loadmat('../data/digits/ex4data1.mat')
    X = training_data.get("X")
    y = training_data.get("y")
    targets = []
    for j in y:
        t = [0] * 10
        t[j-1] = 1
        targets.append(t)
        
    def evaluate(nn, inputs, targets):
        wrong = 0
        right = 0
        for jindex, x in enumerate(inputs):
            p = nn.predict(x)
            maxind = p.argmax() + 1
            if maxind == y[jindex]:
                right += 1
            else:
                wrong += 1
        #print "right: %s, wrong: %s" % (right, wrong)
        acc = right / float(len(y))
        return acc
        
    findparameters.find(evaluate, X, targets, net_type="feedforward", input_size=400, output_size=10, 
                        output_activation="sigmoid", cost_function="ce")
 

You should get accuracy for a bunch of different hyperparameters configurations, some of them:
 
::
 
	hidden_size: 250, activation: tanh, alpha: 0.1, lambda: 0, iter: 1, batch_size: 5 ---- 0.9104
	hidden_size: 250, activation: tanh, alpha: 0.1, lambda: 0, iter: 1, batch_size: 50 ---- 0.9292
	hidden_size: 250, activation: tanh, alpha: 0.1, lambda: 0, iter: 5, batch_size: 5 ---- 0.9784
	hidden_size: 250, activation: tanh, alpha: 0.1, lambda: 0, iter: 5, batch_size: 50 ---- 0.9878
	hidden_size: 250, activation: tanh, alpha: 0.1, lambda: 0, iter: 10, batch_size: 5 ---- 0.9994
	




