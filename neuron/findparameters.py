from neuralnet import NN
from recurrent import Recurrent

#alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
alpha = [0.1]
#lamda = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
lamda = [0]
batch_size = [5, 50]
#hidden_size = [10, 50, 100, 250]
hidden_size = [250]
#activation = ["sigmoid", "tanh"]
activation = ["tanh"]
#iterations = [1, 5, 10]
iterations = [10]

def find(evaluate_func, inputs, targets, net_type, input_size, output_size, output_activation, cost_function):
    for h_size in hidden_size:
        for act in activation:
            if net_type == "feedforward":
                nn = NN([input_size, h_size, output_size], [act, output_activation], cost_function)
            elif net_type == "recurrent":
                nn = Recurrent([input_size, h_size, output_size], [act, output_activation], cost_function)
            for a in alpha:
                for l in lamda:
                    for it in iterations:
                        if net_type == "feedforward":
                            for b_size in batch_size:
                                nn.train(inputs, targets, b_size, alpha=a, lamda=l, iterations=it)
                                v = evaluate_func(nn, inputs, targets)
                                print "hidden_size: %s, activation: %s, alpha: %s, lambda: %s, iter: %s, batch_size: %s ---- %s" % (h_size, act, a, l, it, b_size, v) 
                        elif net_type == "recurrent":
                            b_size = 1
                            try:
                                nn.train(inputs, targets, b_size, alpha=a, lamda=l, iterations=it)
                                v = evaluate_func(nn, inputs, targets)
                                print "hidden_size: %s, activation: %s, alpha: %s, lambda: %s, iter: %s, batch_size: %s ---- %s" % (h_size, act, a, l, it, b_size, v) 
                            except Exception as e:
                                print "Exception when searching for hyperparameters:"
                                print e
        
