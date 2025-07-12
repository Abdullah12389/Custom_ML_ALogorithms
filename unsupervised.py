import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

class Kmeans_Clusterring:
    def __init__(self,k,epochs):
        self.clusters={}
        for i in range(k):
            self.clusters[f"{i}"]=[]
        self.epochs=epochs
        self.k=k
        self.points=None
    def fit(self,xtrain):
        self.points=xtrain[np.random.choice(xtrain.shape[0],self.k,replace=False)]
        for _ in range(self.epochs):
            for i,row in enumerate(xtrain):
                min_index=np.array([np.linalg.norm(row-point) for point in self.points]).argmin() #gets the index at which the point has minimum distance with row
                self.clusters[f"{min_index}"].append({f"{i}":row})
            #pint is a list of points with thrie indexes in array
            #value is dict of point_index and the array of the point 
            self.points=np.array([np.mean(np.array([list(value.values())[0] for value in pints]),axis=0) for pints in self.clusters.values()])
    def predict_cluster(self,points):
        distances=[]
        for point in points:
            min_distant=np.array([np.linalg.norm(point-pint) for pint in self.points]).argmin()
            distances.append(min_distant)
        return np.array(distances).reshape(-1,1)



data=make_blobs(n_samples=100,n_features=2)[0]
model=Kmeans_Clusterring(3,10)
model.fit(data)
for cluster in model.clusters.values(): 
    cluster_points=pd.DataFrame(np.array([list(value.values())[0] for value in cluster]))
    plt.scatter(x=cluster_points[0],y=cluster_points[1])
plt.show()


