import scipy.io
from neuron import findparameters
import time

def start():
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
        
    
if __name__ == "__main__":
    start_time = time.time()
    start()
    end = time.time()
    print "duration: %s" % (end - start_time)
            
            