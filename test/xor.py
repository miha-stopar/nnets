from neuron.neuralnet import NN
import time

if __name__ == "__main__":
    start_time = time.time()
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    targets = [[0], [1], [1], [0]]
    nn = NN([2, 2, 1], ["sigmoid", "sigmoid"], cost_function="ce")
    nn.train(inputs, targets, batch_size=4, alpha=1, lamda=0.0, iterations=3000)
    preds = []
    for index, inp in enumerate(inputs):
        pred = nn.predict(inp)
        preds.append(pred)
        print "%s -> %s" % (inp, pred)
    end = time.time()
    print "duration: %s" % (end - start_time)
    assert preds[0] < 0.01 and preds[1] > 0.99 and preds[2] > 0.99 and preds[3] < 0.01
    
