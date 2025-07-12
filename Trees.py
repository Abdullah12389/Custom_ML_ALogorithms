import numpy as np
import pandas as pd
import grad_losses as lsgd

class BinaryTree:
    depth_reached=0
    def __init__(self,x,y,categories,func,depth,typeof):
        BinaryTree.depth_reached+=1
        self.colname=None
        self.threshold=None
        if typeof=="cls":
            self.categories=y.value_counts().reindex(categories,fill_value=0)
        else:
            self.mean=y.mean()
        self.leftchild=None
        self.rightchild=None
        if len(set(y))>1 and BinaryTree.depth_reached<depth:
            leftx,lefty,rightx,righty=self.build_tree(x,y,func)
            self.leftchild=BinaryTree(leftx,lefty,categories,func,depth,typeof)
            self.rightchild=BinaryTree(rightx,righty,categories,func,depth,typeof)
    def build_tree(self,x,y,func):
        xy=pd.concat([x,y],axis=1)
        self.threshold,self.colname=func(x,y)
        df_left=xy[xy[self.colname]>self.threshold]
        leftx=df_left.iloc[:,:-1]
        lefty=df_left.iloc[:,-1]

        df_right=xy[xy[self.colname]<=self.threshold]
        rightx=df_right.iloc[:,:-1]
        righty=df_right.iloc[:,-1]
        return leftx,lefty,rightx,righty   
      
class Decisiontree_Classifier:
    def __init__(self,x,y,typeof="simple",sample_size=1,depth=999999999):
        self.depth=depth
        if typeof=="random":
            self.sample_size=sample_size
        self.typeof=typeof
        self.tree=BinaryTree(x,y,pd.Series(y).unique(),self.fit,self.depth,typeof="cls")
    def fit(self,xtrain,ytrain):
        categories=pd.Series(ytrain).unique()
        length=len(ytrain)
        gini_impurity_of_columns=[]
        column_threshold=[]
        for col in xtrain:
            gini_impurity_of_threshold=[]
            thresholds=[]
            temp_sorted=xtrain.sort_values(by=col)
            if self.typeof=="random":
                temp_sorted=temp_sorted.sample(n=self.sample_size,replace=False,axis=1)
            unique_col=temp_sorted[col].unique()
            for i,value in enumerate(unique_col[:-1]):
                threshold=(value+unique_col[i+1])/2
                thresholds.append(threshold)
                left=ytrain[temp_sorted[col]>threshold]
                leftimpurity=1-((left.value_counts().reindex(categories,fill_value=0)/len(left))**2).sum()
                right=ytrain[temp_sorted[col]<=threshold]
                right_impurity=1-((right.value_counts().reindex(categories,fill_value=0)/len(right))**2).sum()
                weight_left=len(left)/length
                weight_right=len(right)/length
                weighted_impurity=(weight_left*leftimpurity+weight_right*right_impurity)/2
                gini_impurity_of_threshold.append(weighted_impurity)
            min_index=np.array(gini_impurity_of_threshold).argmin()
            split_threshold=thresholds[min_index]
            column_threshold.append(split_threshold)
            gini_impurity_of_columns.append(np.array(gini_impurity_of_threshold).min())
        minimum_impurity_index_in_columns=np.array(gini_impurity_of_columns).argmin()
        threshold_of_split_column=column_threshold[minimum_impurity_index_in_columns]
        column=xtrain.columns[minimum_impurity_index_in_columns]
        return threshold_of_split_column,column
    def predict(self,xtest):
        def way_down_we_go(tree,row):
            if tree.colname!=None:
                if row[tree.colname]>tree.threshold:
                    return way_down_we_go(tree.leftchild,row)
                else:
                    return way_down_we_go(tree.rightchild,row)
            else:
                return tree.categories.idxmax()
        l=[]
        for i in range(len(xtest)):
            row=xtest.iloc[i]
            l.append(way_down_we_go(self.tree,row))
        return np.array(l).reshape(-1,1)

class DecisionTree_Regressor:
    def __init__(self,x,y,typeof="simple",sample_size=1,depth=999999999):
        self.typeof=typeof
        if typeof=="random":
            self.sample_size=sample_size
        self.depth=depth
        self.tree=BinaryTree(x=x,y=y,categories=pd.Series(y).unique(),func=self.fit,depth=self.depth,typeof="reg")
    def fit(self,xtrain,ytrain):
        length=len(ytrain)
        mse_of_columns=[]
        column_threshold=[]
        for col in xtrain:
            temp_sorted=xtrain.sort_values(by=col)
            if self.typeof=="random":
                temp_sorted=temp_sorted.sample(n=self.sample_size,replace=False,axis=1)
            split_threshold=[]
            split_mse=[]
            for i,value in enumerate(temp_sorted[col][:-1]):
                threshold=(value+temp_sorted[col][i+1])/2
                left_df=ytrain[xtrain[col]>threshold]
                left_mse=lsgd.mse(left_df,left_df.mean())
                weghted_left=left_mse*(len(left_df)/length)
                right_df=ytrain[xtrain[col]<=threshold]
                right_mse=lsgd.mse(right_df,right_df.mean())
                weghted_right=right_mse*(len(right_df)/length)
                average_mse=(weghted_right+weghted_left)/2
                split_threshold.append(threshold)
                split_mse.append(average_mse)
            min_index=np.array(split_mse).argmin()
            column_threshold.append(split_threshold[min_index])
            mse_of_columns.append(split_mse[min_index])
        min_index=np.array(mse_of_columns).argmin()
        return column_threshold[min_index],xtrain.columns[min_index]
    def predict(self,xtest):
        def way_down_we_go(tree,row):
            if tree.colname!=None:
                if row[tree.colname]>tree.threshold:
                    return way_down_we_go(tree.leftchild,row)
                else:
                    return way_down_we_go(tree.rightchild,row)
            else:
                return tree.mean
        l=[]
        for i in range(len(xtest)):
            row=xtest.iloc[i]
            l.append(way_down_we_go(self.tree,row))
        return np.array(l).reshape(-1,1)
    
