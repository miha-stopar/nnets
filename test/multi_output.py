from neuron.neuralnet import NN
import time

if __name__ == "__main__":
    start_time = time.time()
    inputs = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    targets = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    nn = NN([3, 3, 5, 3], ["sigmoid", "tanh", "softmax"], cost_function="softmax_ce")
    nn.train(inputs, targets, batch_size=4, alpha=1, lamda=0.0, iterations=1000)
    preds = []
    for index, inp in enumerate(inputs):
        pred = nn.predict(inp)
        preds.append(pred)
        print "%s -> %s" % (inp, pred)
    end = time.time()
    e1 = sum(abs(preds[0] - targets[0]))
    e2 = sum(abs(preds[1] - targets[1]))
    e3 = sum(abs(preds[2] - targets[2]))
    print "%s   %s   %s" % (e1, e2, e3)
    print "duration: %s" % (end - start_time)
    assert e1 < 0.006 and e2 < 0.006 and e3 < 006
    
