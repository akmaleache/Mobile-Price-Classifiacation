# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:44:24 2019

@author: Akmal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


mob_data = pd.read_csv('E:\\akmal\\python data sci\\projects\\mobile data\\mobile-price-classification\\train.csv')
mob_data.dtypes
#univariate
description = mob_data.describe()


#multivariate
correlation_mat = mob_data.corr()
plt.figure(figsize = [10,8])
sns.heatmap(correlation_mat)

#we see relation between px height and width
sns.scatterplot(mob_data['px_height'],mob_data['px_width'])

#converting height and weidth to inches
mob_data['sc_h'] = mob_data['sc_h'] * 0.393701
mob_data['sc_w'] = mob_data['sc_w'] * 0.393701
#feature engineering
mob_data['screen_size'] = np.sqrt(np.square(mob_data['sc_h']) + np.square(mob_data['sc_w']))
mob_data['resolution'] = (mob_data['px_height'] * mob_data['px_width'])
#removing old variable
dcol = ['sc_h','sc_w','px_height','px_width']
mob_data.drop(dcol,axis = 1,inplace = True)
sns.barplot(x = mob_data['price_range'], y = mob_data['screen_size'])
sns.barplot(x = mob_data['price_range'], y = mob_data['resolution'])

sns.FacetGrid(mob_data,hue = 'price_range').map(sns.distplot,'resolution').add_legend();
#sns.pairplot(mob_data.iloc[:,[9,10,11,12,13,14,15,20]])
plt.hist(np.square(mob_data['battery_power']))
plt.boxplot(mob_data['battery_power'])



#for col in mob_data.columns:
#    pd.pivot_table(mob_data, values = col, index = 'price_range')
    

mob_data['blue'].value_counts()
x = list(mob_data.columns)
y = x.pop(16)

def scale(a):
    return (a - np.min(a))/(np.max(a) - np.min(a))

norm_data = mob_data[x].apply(lambda a: scale(a))
norm_data[y] = mob_data[y]

from sklearn.model_selection import train_test_split
train,test = train_test_split(norm_data,train_size = 0.8,test_size = 0.2,stratify = mob_data['price_range'])


train['price_range'].value_counts()
test['price_range'].value_counts()

def built_model(algo,pre):
    algo.fit(train[pre],train[y])
    pred = algo.predict(test[pre])
    print(accuracy_score(test[y],pred))


from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier 
model1 = DecisionTreeClassifier(criterion = 'entropy')
built_model(model1,x)#86.25%
score = model1.feature_importances_
plt.bar(x,score)
filter_x = ['battery_power','ram','resolution']
built_model(model1,filter_x)#89.5%

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators = 200)
built_model(model2,x)#87.75
built_model(model2,filter_x)#91%

#insigths from above features 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

df_fill_x = norm_data[filter_x]
k = list(range(2,8))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_fill_x)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_fill_x.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_fill_x.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

#building model for cluster 4
k_model = KMeans(n_clusters = 4)
k_model.fit(df_fill_x)
k_model.labels_

#crete df with original values
clus_df = mob_data[filter_x]
clus_df['clusters'] = k_model.labels_
pd.pivot_table(clus_df, values = ['battery_power','ram','resolution'], index = 'clusters')
df_fill_x.drop('clusters',axis = 1,inplace = True)
