import scipy.io
from neuron.neuralnet import NN
import time

def start():
    nn = NN([400, 250, 10], ["tanh", "sigmoid"], cost_function="ce")
    training_data = scipy.io.loadmat('../data/digits/ex4data1.mat')
    X = training_data.get("X")
    y = training_data.get("y")
    targets = []
    for j in y:
        t = [0] * 10
        t[j-1] = 1
        targets.append(t)
    nn.train(X, targets, batch_size=5, alpha=0.1, lamda=0.0, iterations=10)
    print "training finished"
    
    wrong = 0
    right = 0
    for jindex, x in enumerate(X):
        p = nn.predict(x)
        maxind = p.argmax() + 1
        if maxind == y[jindex]:
            right += 1
        else:
            wrong += 1
    print "right: %s, wrong: %s" % (right, wrong)
    acc = right / float(len(y))
    return acc
            
if __name__ == "__main__":
    start_time = time.time()
    acc = start()
    print "accuracy: %s" % acc
    end = time.time()
    print "duration: %s" % (end - start_time)
    assert acc > 0.99
            
            