import numpy as np
import Gradient_Descent
import pandas as pd
import grad_losses as lsgd

class Linear_Regression:
    def __init__(self):
        self.weights=None
    def fit(self,xtrain,ytrain):
        x=np.append(np.ones((xtrain.shape[0],1)),xtrain,axis=1)
        self.weights=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),ytrain)
    def predict(self,xtest):
        x=np.append(np.ones((xtest.shape[0],1)),xtest,axis=1)
        return np.dot(x,self.weights)
    
class SGD_Classifier:
    def __init__(self,lr,epochs,features):
        self.gradient_descent=Gradient_Descent.SGD(lr,features,epochs)
    def fit(self,xtrain,ytrain):
        self.gradient_descent.fit(xtrain,ytrain,lsgd.binary_cross_entropy_grad)
    def predict(self,xtest):
        xones=np.append(np.ones((xtest.shape[0],1)),self.scaler.transform(xtest),axis=1)
        return lsgd.sigmoid(np.dot(xones,self.gradient_descent.weights))