import numpy as np
import tools

class NN(object):
    def __init__(self, units_per_layer, activation_functions, cost_function="ce"):
        # use cross entropy cost function (CE) only when output is between 0 and 1 (if sigmoid is last activation)
        # units_per_layer for example : [2,2,1]
        # activation_functions for example : ["tanh", "sigmoid"]
        # when output is one-of-n vector, softmax can be used as output activation
        np.seterr(all = 'raise')
        np.random.seed(0)
        self.units_per_layer = units_per_layer
        self.layers = len(units_per_layer)
        self.weights = []
        self.z = {}
        self.a = {}
        self.activation_functions = activation_functions
        self.activations = []
        self.activations_der = [] # derivatives
        if activation_functions[-1] not in ["sigmoid", "softmax"] and cost_function == "ce":
            assert False, "cross entropy cost function can't be applied for activation output function with range different than [0, 1]"
        if activation_functions[-1] == "softmax" and cost_function != "softmax_ce":
            assert False, "softmax works only with cross-entropy because of back_propagate implementation (some computation is omitted because things cancel each other out - but holds only for CE"
        for a in activation_functions:
            if a == "sigmoid":
                self.activations.append(self.sigmoid)
                self.activations_der.append(self.sigmoid_der)
            elif a == "tanh":
                self.activations.append(np.tanh)
                self.activations_der.append(self.tanh_der)
            elif a == "linear":
                self.activations.append(lambda x : x)
                self.activations_der.append(lambda x : 1)
            elif a == "softmax":
                self.activations.append(self.softmax)
                self.activations_der.append(lambda x : 1)
        self.cost_function_name = cost_function
        if cost_function == "ce":
            self.cost_function_der = self.cost_derivative_ce
        elif cost_function == "sse":
            self.cost_function_der = self.cost_derivative_sse
        for idx in range(1, self.layers):
            fan_in = self.units_per_layer[idx-1] + 1
            fan_out = self.units_per_layer[idx]
            lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))
            weight = np.random.uniform(-lim, lim, [fan_out, fan_in]) 
            self.weights.append(weight)
   
    def feed_forward(self, weights, batch_inputs):
        self.z = {}
        self.a = {}
        bias_inp = np.hstack((np.ones((len(batch_inputs), 1)), batch_inputs))
        self.a[1] = bias_inp
        for ind, weight in enumerate(weights):
            self.z[ind+2] = np.dot(self.a[ind+1], weight.T)
            self.z[ind+2] = np.clip(self.z[ind+2], -50, 50) # for numerical stability
            if ind == len(weights) - 1:
                self.a[ind+2] = self.activations[ind](self.z[ind+2])
            else:
                self.a[ind+2] = np.hstack((np.ones((len(self.z[ind+2]), 1)), self.activations[ind](self.z[ind+2])))
        return self.a[self.layers]
    
    def back_propagate(self, targets):
        delta = {} # errors for each layer
        if (self.activation_functions[-1] == "sigmoid" and self.cost_function_name == "ce") or\
                self.activation_functions[-1] == "softmax":
            # sigmoid derivative and cost function derivative cancel each other out in this case - this if block is
            # just to avoid unnecessary computation - it could be calculated as well in else block
            # similar for softmax
            delta[self.layers] = (self.a[self.layers] - targets)
        else:
            deriv = self.activations_der[-1](self.a[self.layers])
            delta[self.layers] = self.cost_function_der(self.a[self.layers], targets) * deriv
        for idx in range(self.layers-1, 1, -1):
            weight = self.weights[idx-1]
            deriv = self.activations_der[idx-2](self.a[idx])
            cdelta = np.dot(delta[idx+1], weight) * deriv
            cdelta = cdelta[:, 1:]
            delta[idx] = cdelta
        Delta = {}
        for idx in range(1, self.layers):
            fan_in = self.units_per_layer[idx-1] + 1
            fan_out = self.units_per_layer[idx]
            Delta[idx+1] = np.zeros((fan_out, fan_in))
        for idx in range(1, self.layers):
            Delta[idx+1] = np.dot(delta[idx+1].T, self.a[idx])
        return Delta

    def train(self, uinputs, utargets, batch_size=1, alpha=1, lamda=0.0, iterations = 1000, shuffle=True, 
              calculate_errors=False):
        self.calculate_errors = calculate_errors
        l = range(len(uinputs))
        if shuffle:
            inputs = []
            targets = []
            np.random.shuffle(l)
            for i in l:
                if type(uinputs[i]) == np.ndarray:
                    inputs.append(uinputs[i].tolist()) # to convert int32 into int and enable json.dumps
                else:
                    inputs.append(uinputs[i])
                targets.append(utargets[i])
        else:
            inputs = uinputs
            targets = utargets
        self.lamda = lamda
        self.inputs = inputs
        self.targets = targets
        num_of_batches = len(inputs) / batch_size
        if len(inputs) % batch_size != 0:
            num_of_batches += 1
        errors = []
        for _ in range(iterations):
            iter_error = 0
            for i in range(num_of_batches):
                batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
                batch_targets = targets[i*batch_size:(i+1)*batch_size]
                cost_gradient, error = self.cost_gradient(batch_inputs, batch_targets)
                iter_error += error
                self.weights = [self.weights[idx] - alpha*cost_gradient[idx] for idx in range(len(self.weights))]
            errors.append(iter_error)
        return errors
                
    def _prevent_divide_error(self, pred):
        # happens when predicted value contains 1, which leads into -inf for np.log(1-pred)
        # in this case multiplication with (1-np.array(batch_targets) would 
        # change this element into 0 anyway, so we can change this element into whatever
        t = 1 - pred
        for ind, pr in enumerate(t):
            if type(pr) == list or type(pr) == np.ndarray:
                for jnd, p in enumerate(pr):
                    if p == 0:
                        pr[jnd] = 0.1 # something
            else:
                if pr == 0:
                    t[ind] = 0.1 # something 
        part = np.log(t)
        return part
    
    def cost_gradient(self, batch_inputs, batch_targets):
        pred = self.feed_forward(self.weights, batch_inputs)
        cost = 0
        if self.calculate_errors:
            if self.cost_function_name == "sse":
                try:
                    cost = sum(sum(np.power(batch_targets - pred, 2)))
                except:
                    print "exception when calculating cost_gradient (sse)"
            else:
                try:
                    part = np.log(1-pred)
                except:
                    part = self._prevent_divide_error(pred)
                    print "exception when calculating cost gradient; pred: %s" % pred
                cost = sum(sum(-np.array(batch_targets) * np.log(pred) - (1-np.array(batch_targets)) * part))
            cost = cost / float(len(batch_targets))
            if self.cost_function_name == "sse":
                cost = cost/float(2)
            #print cost
        Delta = self.back_propagate(batch_targets)
        cost_gradient = []
        m = float(len(batch_targets))
        for i in range(len(Delta)):
            if self.lamda > 0:
                Delta[i+2] = (1/m)*Delta[i+2] + (1/m)*self.lamda * np.hstack((np.zeros((self.weights[i].shape[0], 1)), 
                                                                              self.weights[i][:, 1:]))
            else:
                Delta[i+2] = (1/m)*Delta[i+2]
            cost_gradient.append(Delta[i+2])
        # uncomment the following line to see if back propagation works ok:
        #self.check_gradient(cost_gradient, batch_inputs, batch_targets)
        return cost_gradient, cost
    
    def cost(self, batch_inputs, batch_targets):
        c = 0
        for index, inp in enumerate(batch_inputs):
            pred = self.feed_forward(self.weights, [inp])[0]
            target = batch_targets[index]
            if self.cost_function_name == "sse":
                d = sum(np.power(target - pred, 2))
            elif self.cost_function_name == "ce":
                try:
                    part = np.log(1-pred)
                except:
                    part = self._prevent_divide_error(pred)
                d = -np.inner(np.array(target), np.log(pred)) - np.inner((1-np.array(target)), part)
            elif self.cost_function_name == "softmax_ce":
                d = -np.inner(np.array(target), np.log(pred))
            c += d
        if self.cost_function_name == "sse":
            c = c/float(2)
        reg = 0 
        m = float(len(batch_inputs))
        c = (1/m)*c
        for weight in self.weights:
            reg = reg + self.lamda/(2*m)*np.power(weight[:, 1:], 2).sum()
        return c + reg
    
    def check_gradient(self, cost_gradient, batch_inputs, batch_targets): 
        # check if gradient is implemented correctly - the difference between cgrad and grad
        # should be < e-9
        # note1: do not call this function when you actually train a neural net - it slows down the execution heavily
        # note2: doesn't work for recurrent net, because hidden layer is copied into output again in "feed_forward(-epsilon)" and "feed_forward(+epsilon)"
        epsilon = 0.0001
        for ind, w in enumerate(self.weights):
            wc = w
            for ir in range(w.shape[0]):
                for jc in range(w.shape[1]):
                    wc[ir, jc] += epsilon
                    cost2 = self.cost(batch_inputs, batch_targets)
                    wc[ir, jc] -= 2*epsilon
                    cost1 = self.cost(batch_inputs, batch_targets)
                    wc[ir, jc] += epsilon # set back to the original value
                    grad = (cost2 - cost1) / (2 * epsilon)
                    cgrad = cost_gradient[ind][ir, jc]
                    diff = abs(cgrad - grad)
                    #print diff
                    if diff > 1e-9:
                        assert False, "not correctly computed gradient, difference: %s" % diff
            
    def sigmoid(self, x):
        try:
            #r = 1 /(1+np.exp(-x))
            r = 1 /(1+tools.safe_exp(-x))
            return r
        except Exception as e:
            print e
            print "exception when calculating sigmoid"
        
    def softmax(self, x):
        # rows of x are expected to be vectors where softmax is to be applied
        r = np.copy(x)
        for index, col in enumerate(x):
            #m = np.max(col)
            #r[index] = np.exp(col - m)
            #r[index] = tools.safe_exp(col - m)
            r[index] = tools.safe_exp(col)
            r[index] = r[index] / np.sum(r[index])
        return r
    
    def tanh_der(self, x):
        der = 1 - x * x
        return der

    def sigmoid_der(self, x):
        der = x * (1 - x)
        return der
    
    def cost_derivative_ce(self, a, t):
        # Cross-entropy tends to allow errors to change weights even when nodes saturate 
        # (which means that their derivatives are asymptotically close to 0.)
        return (a - t) / (a * (1 - a)) 
    
    def cost_derivative_sse(self, a, t):
        # sum squared error
        # t - target value, a - calculated (feed_forwarded) value
        return (a - t)
    
    def predict(self, inp):
        pred = self.feed_forward(self.weights, [inp])
        return pred[0]
    
