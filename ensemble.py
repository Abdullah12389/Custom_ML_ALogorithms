import Trees
import pandas as pd
import numpy as np

class RandomForest:
    def __init__(self,max_depth,typeof,n_estimators,sample_column=3,sample_row=3):
        self.estimators=n_estimators
        self.depth=max_depth
        self.sample_column=sample_column
        self.sample_row=sample_row
        self.typeof=typeof
        if typeof not in ["reg","cls"]:
            raise AttributeError("not a valid attribute use reg for regression or cls for classification")
        self.trees=[]
    def fit(self,xtrain,ytrain):
        column_split=int(np.ceil(xtrain.shape[1]/self.sample_column)) #this will make the sample taken from dataset=1 if we get less columns than split factor
        row_split=int(np.ceil(xtrain.shape[0]/self.sample_row))  #this will make the sample taken from dataset=1 if we get less rows than split factor
        for _ in range(len(self.estimators)):
            x=xtrain.sample(n=row_split,replace=True)
            y=ytrain.iloc[x.index]
            if self.type_of=="cls":
                dt=Trees.Decisiontree_Classifier(x,y,"random",sample_size=column_split,depth=self.depth)
            elif self.typeof=="reg":
                dt=Trees.DecisionTree_Regressor(x,y,"random",sample_size=column_split,depth=self.depth)
            self.trees.append(dt)
    def predict(self,xtest):
        pred_rows=[]
        for row in xtest.values:
            l=[tree.predict(pd.DataFrame(row.reshape(1,-1),columns=xtest.columns)) for tree in self.trees]
            pred_rows.append(np.mean(l))
        return np.array(pred_rows).reshape(-1,1)

class GradientBoost:
    def __init__(self,n_estimators,max_depth,typeof,learning_rate):
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.trees=[]
        self.typeof=typeof
        self.mean_pred=None
        self.lr=learning_rate
    def fit(self,xtrain,ytrain):
        self.mean_pred=np.mean(ytrain)
        residual=ytrain-np.mean(ytrain)
        for _ in range(len(self.n_estimators)):
            if self.typeof=="reg":
                dt=Trees.DecisionTree_Regressor(xtrain,residual,depth=self.max_depth)
            else:
                dt=Trees.Decisiontree_Classifier(xtrain,residual,depth=self.max_depth)
            pred=dt.predict(xtrain)
            residual=residual-self.lr*pred
            self.trees.append(dt)
    def predicts(self,xtest):
        pred_rows=[]
        for row in xtest.values:
            sum_of_trees=np.array([tree.predict(pd.DataFrame(row.reshape(1,-1),columns=xtest.columns)) for tree in self.trees]).sum()
            pred=self.mean_pred+self.lr*sum_of_trees
            pred_rows.append(pred)
        return np.array(pred_rows).reshape(-1,1)
    