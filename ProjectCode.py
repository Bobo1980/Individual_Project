# -*- coding: utf-8 -*-
'''
Created on Thu Oct 29 18:57:54 2015

@author: 
'''

from bs4 import BeautifulSoup as bs
import requests as rq
from pandas.io import wb
import pandas as pd
from pandas import DataFrame
import MySQLdb as mySQL
import pandas.io.sql as pdSQL
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
import rpy2.robjects.lib.ggplot2 as ggplot2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from IPython.core.display import Image
import warnings
import pandas.rpy.common as com
#import wbdata
import numpy as np

import scipy.spatial.distance as dist
import scipy
import matplotlib.pyplot as plt # 2D plotting

from sklearn.metrics import euclidean_distances 
from sklearn.metrics.pairwise import linear_kernel as cosine_distances 
from sklearn.metrics.pairwise import manhattan_distances as manhattan_distances 
from sklearn import manifold  # multidimensional scaling
from sklearn import decomposition , preprocessing
from scipy.cluster.vq import kmeans,vq 
from pylab import plot,show
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
 


indicators = ['EG.ELC.ACCS.ZS', 'NY.GDP.PCAP.CD', 'EG.GDP.PUSE.KO.PP.KD']

# get data from world bank API using indicators
def get_data(indi):
    download = wb.download(indicator = indi, country = 'all', start = 2012, end = 2012)
    first_34 = download[34:]
    
    return first_34

# get access to electricity data
accToElec = get_data('EG.ELC.ACCS.ZS')
# get aGDPdata
GDP = get_data('NY.GDP.PCAP.CD')
# get energy per GDP unit data
energy_per_GDPunit = get_data('EG.GDP.PUSE.KO.PP.KD')

#Put data in one data frame
access_GDP = DataFrame.merge(accToElec, GDP, left_index=True, right_index=True)

df = DataFrame.merge(access_GDP, energy_per_GDPunit, left_index=True, right_index=True)

df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

df.reset_index(inplace = True)
#Assign column names
df.columns=["Country", "Year", "Access_to_Electricity", "GDP", "Energy_per_GDPunit"]
#Convert to a csv file
df.to_csv("df.csv")


  
#Convert data frame to sql function
def df_to_mysql(dbName, tableName, df):
    conn = mySQL.connect(host='localhost', user='root', passwd='root')
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS " + dbName + ";")      
    conn = mySQL.connect(host='localhost', user='root', passwd='root', db=dbName)
    df.to_sql(tableName, conn, flavor='mysql', if_exists='replace', index=False)
    cursor = conn.cursor()
    cursor.execute(' USE %s; ' % (dbName) ) 
##   UNCOMMENT TO SEE SQL TABLE DATA
#    myDataFrame = pdSQL.read_sql('SELECT * FROM %s' % (tableName), conn)
#    print myDataFrame
    conn.close()

df_to_mysql('individualProj', 'Electricity_Access_and_Usage_GDP', df)

# Convert SQL file back to data frame
def mysql_to_df(dbName, tableName):
    conn = mySQL.connect(host='localhost', user='root', passwd='root', db=dbName)
    df = pd.read_sql('SELECT * FROM %s;' % (tableName), con=conn)  
    return df
    conn.close()

Data = mysql_to_df('individualProj', 'Electricity_Access_and_Usage_GDP')
#Convert to R file for plotting
rData= com.convert_to_r_dataframe(Data)

def plot(data, x, y, ylabel, color, filename):
    gp = ggplot2.ggplot(data=data)
    gp = gp + \
    ggplot2.geom_line(ggplot2.aes_string(x=x, y=y), color=color) + \
    ggplot2.theme(**{'axis.text.x' : ggplot2.element_text(angle = 90, hjust = 1),
                      'strip.text.y' : ggplot2.element_text(size = 6, angle=90)})  + \
    ggplot2.scale_y_continuous(ylabel) 
    ggplot2.ggplot2.ggsave(filename, gp)
    
plot(rData, 'GDP', 'Access_to_Electricity', "Electricity access in %", 'steelblue', 'GDP_vs_Access_to_Electricity.png')
plot(rData, 'GDP', 'Energy_per_GDPunit', "Energy per GDP Unit", 'red', 'GDP_vs_Energy_Per_Unit.png')
plot(rData, 'Access_to_Electricity', 'Energy_per_GDPunit', "Energy per GDP Unit", 'green', 'Electricity_vs_Energy.png')



# Cluster analysis plot
data = Data.ix[:,'Access_to_Electricity':]

df2 = preprocessing.scale(data)

# Perform principal components analysis
pca = decomposition.PCA()
pca.fit(df2)

#print(pca.explained_variance_) 

# Prompt the user to enter the number of K-means of choice


# Get the first 2 principal components
pca.n_components = 2
df_reduced = pca.fit_transform(df2)
df_reduced.shape # Check the shape of the data
# The next code segments show how to # obtain plots for different number # of clusters. Run each segment  # separately to see the results # computing K-Means with K =2 (2 clusters)

#centroids,_ = kmeans(df_reduced,2)

# computing K-Means with K = 2 (2 clusters)
#centroids,dist =kmeans(df_reduced,n) 
centroids,_ = kmeans(df_reduced,5)
# assign each sample to a cluster
idx,_ = vq(df_reduced,centroids)

# some plotting using numpy's logical indexing
colors = cm.rainbow(np.linspace(0, 1, 5))
data_colors = colors[idx]


for i in range(0,5):

    plt.scatter(df_reduced[idx==i,0],df_reduced[idx==i,1],c=data_colors[idx==i],s=30) 
    

plt.scatter(centroids[:,0],centroids[:,1],c='yellow',s=30, marker='s') 
show() 





