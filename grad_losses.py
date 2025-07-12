import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z)) 

def binary_cross_entropy_grad(xy,weights):
     x=xy[:,:-1]
     y=xy[:,-1].reshape(len(xy),1)
     yhat=sigmoid(np.dot(x,weights))
     return np.dot(x.T,(yhat-y))/len(x)

def mse_grad(xy,weight):
     x=xy[:,:-1]
     y=xy[:,-1]
     loss=y-np.dot(x,weight)
     return -2*np.dot(x.T,loss)

def mse(y,yhat):
     return ((y-yhat)**2)/len(y)
