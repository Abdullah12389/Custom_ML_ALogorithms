import numpy as np

class SGD:
    def __init__(self,lr,columns,epochs):
        self.weights=np.random.random((columns+1,1))
        self.lr=lr
        self.epochs=epochs
    def fit(self,xtrain,ytrain,grad_loss):
        x=np.concatenate((np.ones((xtrain.shape[0],1)),xtrain,ytrain),axis=1)
        for _ in range(self.epochs):
            np.random.shuffle(x)
            for row in x:
                weight_grad=grad_loss(row.reshape(1,row.shape[0]),self.weights)
                self.weights=self.weights-self.lr*weight_grad
        return self.weights
    
class BatchGD:
    def __init__(self,lr,columns):
        self.weights=np.random.random((columns+1,1))
        self.lr=lr
        self.bias=np.random.random((1,1))
    def fit(self,xtrain,ytrain,grad_loss):
        xy=np.concatenate((np.ones((xtrain.shape[0],1)),xtrain,ytrain),axis=1)
        weight_grad=grad_loss(xy,self.weights)
        self.weights=self.weights-self.lr*weight_grad
        return self.weights
    
class MiniBatchGD:
    def __init__(self,lr,columns,epochs,batch_size):
        self.weights=np.random.random((columns+1,1))
        self.lr=lr
        self.bias=np.random.random((1,1))
        self.batch_size=batch_size
        self.epochs=epochs
    def fit(self,xtrain,ytrain,grad_loss):
        xy=np.concatenate((np.ones((xtrain.shape[0],1)),xtrain,ytrain),axis=1)
        for i in range(self.epochs):
            np.random.shuffle(xy)
            for i in range(0,xtrain.shape[0],self.batch_size):
                batch=xy[i:i+self.batch_size]
                grad=grad_loss(batch,self.weights)
                self.weights=self.weights-self.lr*grad
        return self.weights