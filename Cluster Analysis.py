# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:57:30 2015

@author: Wendy Zhang G31476912

This program reads a csv file and performs a cluster analysis of users based on their preferences
for the list of items provided.

"""

import pandas as pd 
from pandas import ExcelWriter 
from pandas import ExcelFile 
import numpy as np
from sklearn import decomposition , preprocessing
from scipy.cluster.vq import kmeans,vq 
from pylab import plot,show
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

readin_data = pd.read_csv('items.csv') 
readin_data= readin_data.astype('float')

# This is a recommended step to normalize 
# variables to a common scale metric
df2 = preprocessing.scale(readin_data)

# Perform principal components analysis
pca = decomposition.PCA()
pca.fit(df2)

#print(pca.explained_variance_) 

# Prompt the user to enter the number of K-means of choice
n = input("Input an integer for number of K-Means > ")

# Get the first 2 principal components
pca.n_components = 2
df_reduced = pca.fit_transform(df2)
df_reduced.shape # Check the shape of the data
# The next code segments show how to # obtain plots for different number # of clusters. Run each segment  # separately to see the results # computing K-Means with K =2 (2 clusters)

#centroids,_ = kmeans(df_reduced,2)

# computing K-Means with K = 2 (2 clusters)
#centroids,dist =kmeans(df_reduced,n) 
centroids,_ = kmeans(df_reduced,n)
# assign each sample to a cluster
idx,_ = vq(df_reduced,centroids)

# some plotting using numpy's logical indexing
colors = cm.rainbow(np.linspace(0, 1, n))
data_colors = colors[idx]


for i in range(0,n):
    #plot(df_reduced[idx==i,0],df_reduced[idx==i,1],c=data_colors)
    plt.scatter(df_reduced[idx==i,0],df_reduced[idx==i,1],c=data_colors[idx==i],s=30) 
    

plt.scatter(centroids[:,0],centroids[:,1],c='yellow',s=30, marker='s') 
show() 


