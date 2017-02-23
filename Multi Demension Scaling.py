# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:01:48 2015

@author: Wendy Zhang G31476912
This program reads a csv file and calculates the distance among the items provided. 

"""
import pandas as pd 
import numpy as np
from pandas import ExcelWriter 
from pandas import ExcelFile 
import scipy.spatial.distance as dist
import scipy
import matplotlib.pyplot as plt # 2D plotting
# alternative distance metrics for multidimensional scaling
from sklearn.metrics import euclidean_distances 
from sklearn.metrics.pairwise import linear_kernel as cosine_distances 
from sklearn.metrics.pairwise import manhattan_distances as manhattan_distances 
from sklearn import manifold  # multidimensional scaling
 
# load the sheet into the Pandas Dataframe structure 
df = pd.read_csv('items.csv')
#df = pd.read_excel('items.xlsx', sheetname='Breakfast Items')

#transpose the data to make sure we are plotting the breakfast items
data = df.transpose()

data= data.astype('float')

##item_data = df
# apply the multidimensional scaling algorithm and plot the map
np.set_printoptions(precision=3) 
distance_matrix= scipy.spatial.distance.squareform(dist.pdist(data,"euclidean"))


mds_method =manifold.MDS(n_components = 2, random_state = 9999,dissimilarity = 'precomputed') 
mds_fit =mds_method.fit(distance_matrix) 
mds_coordinates =mds_method.fit_transform(distance_matrix) 
item_label =['toast pop-up', 'buttered toast', 'English muffin and margarine', 'jelly donut', 'cinnamon toast', 
'blueberry muffin and margarine', 'hard rolls and butter', 'toast and marmalade', 'buttered toast and jelly', 
'toast and margarine', 'cinnamon bun', 'Danish pastry', 'glazed donut', 'coffee cake', 'corn muffin and butter'] 


# plot mds solution in two dimensions using city labels
# defined by multidimensional scaling

plt.figure() 
plt.scatter(mds_coordinates[:,0],mds_coordinates[:,1],facecolors = 'none', edgecolors = 'none') # points in white (invisible)

labels =item_label

for label, x, y in zip(labels, mds_coordinates[:,0], mds_coordinates[:,1]): 
    plt.annotate(label, (x,y), xycoords = 'data') 
    plt.xlabel('Breakfast Items') 
    plt.ylabel('Clusters') 
    plt.show() 
    plt.savefig('fig_positioning_products_mds_cities_python.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='landscape', papertype=None, format=None, 
    transparent=True, pad_inches=5, frameon=None) 