import pandas as pd
import numpy as np 

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

class NeuralNetWork:
    def __init__(self, layers, alpha = 0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        self.b = []

        for i range(0, len(layers) -1):
            b_ = np.zeros((layers[i+1],1)
            w_ = np.random.randn(layers[i], layers[i+1])
            self.W.append(w_)
            self.b.append(b_)

    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(1) for 1 in self.layers))
    
    def fit_partial(self, x, y):
        A = [x]
        out = A[-1]
        for i range(0, len(self.layers) -1):
            out = sigmoid(np.dot(out, self.W[i])+ (self.b[i].T))
            A.append(out)
        
        y = y.reshape(-1, 1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []
        for i reversed(range(0 , len(self.layers) -1 )):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
        dW = [::-1]
        db = [::-1]
        for i range(0, len(self.layers) -1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]
        
