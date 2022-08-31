#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:43:17 2020

@author: tahereh
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans , DBSCAN 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import unique
from numpy import where
from scipy.stats import bernoulli
from pandas import DataFrame



def joinMovieLensIMDB():
    
    movies = pd.read_csv('../data/movies.csv', sep=",")
    movies = movies.drop_duplicates(subset ="title", keep = 'first', inplace = False) 
    imdb_movies = pd.read_csv('../data/IMDb movies.csv', sep=",")
    imdb_movies = imdb_movies.drop_duplicates(subset ="title", keep = 'first', inplace = False) 
    mergedf = pd.merge(movies , imdb_movies , on=['title'], how='inner')
    mergedf['production_company'].replace('', np.nan, inplace=True)
    mergedf.dropna(subset=['production_company'], inplace=True)
    mergedf.to_csv('../data/movies_merged.csv', sep="," , index = False)
    print(len(mergedf))
    print(mergedf.head())

def cleanMovieTitle():
    
    movies = pd.read_csv('../data/movies.csv', sep=",")
    movies['title'] = movies['title'].str.replace('\d+', '')
    movies['title'] = movies['title'].str.replace(' \(\)', '')
    #print(movies['title'].head())
    movies.to_csv('../data/movies.csv', sep="," , index = False)

# only keep the ratings for the remaining movies 
def filterRatings():
    
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    movies_merged = pd.read_csv('../data/movies_merged.csv', sep=",")
    
    new_ratings = ratings[ratings['movieId'].isin(movies_merged['movieId'])]
    print(len(new_ratings))
    new_ratings.to_csv('../data/ratings_filtered.csv', sep="," , index = False)
    
    
def getProductionCompanies():
    
    movies_merged = pd.read_csv('../data/movies_merged.csv', sep=",")
    production_companies = movies_merged['production_company']
    #print(len(production_companies))
    production_companies = production_companies.unique()
    #print(len(production_companies))
    
    productions = pd.DataFrame(columns = ['productionId','production_company']) # plus preferences
    productions['name'] = production_companies
    
    # add unique companies to csv then add columns as their preferences
    productions.to_csv('../data/productions.csv', sep="," )
    

def joinProductionsToMovies():
    
    productions = pd.read_csv('../data/productions.csv', sep=",")
    movies_merged = pd.read_csv('../data/movies_merged.csv', sep=",")
    
    mergedf = pd.merge(movies_merged , productions , on=['production_company'], how='inner')
    mergedf.to_csv('../data/movies_merged.csv', sep="," , index = False)
    #print(len(mergedf))
    
def createSample():
    
    productions = pd.read_csv('../data/productions.csv', sep=",")
    movies = pd.read_csv('../data/movies.csv', sep=",")
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    users = pd.read_csv('../data/users.csv', sep=",")
     
    movies = movies[movies['production_company'].isin(productions['production_company'])]
    movies = movies.loc[(movies['genre'] != '(no genres listed)')]
    
    ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
    
    movies.to_csv('../data/movies_sample.csv', sep="," , index = False)
    ratings.to_csv('../data/ratings_sample.csv', sep="," , index = False)
    
    users = users[users['userId'].isin(ratings['userId'])]
    users.to_csv('../data/users_sample.csv', sep="," , index = False)
    
def create5folds():
    
    movies = pd.read_csv('../data/movies.csv', sep=",")
    users = pd.read_csv('../data/users.csv', sep=",")
    
    movies_shuffled = movies.sample(frac=1)
    movies_result = np.array_split(movies_shuffled, 5)
    i = 1
    for part in movies_result:
        path = '../data/5fold/movies' + str(i) + '.csv'
        part.to_csv(path, sep="," , index = False)
        i = i + 1
        
    users_shuffled = users.sample(frac=1)
    users_result = np.array_split(users_shuffled, 5)
    i = 1
    for part in users_result:
        path = '../data/5fold/users' + str(i) + '.csv'
        part.to_csv(path, sep="," , index = False)
        i = i + 1    


def elbow(data):
    
    """
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)
    """
    Sum_of_squared_distances = []
    K = range(1,30)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)  
           
        
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    

def DBScan_epsilon(data):
    
    neigh = NearestNeighbors(n_neighbors=6) #>=2
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    

def clusterUsers(iskmeans):
    
    userdf = pd.read_csv('../data/users.csv', sep=",")
    occupations = pd.read_csv('../data/occupations.csv', sep=",")
    userdf = pd.merge(userdf , occupations , on=['occupationId'], how='inner')
    
    userdf.loc[((userdf['age'] <= 20)), 'age_group'] = '<=20'
    userdf.loc[((userdf['age'] > 20) & (userdf['age'] <= 30)), 'age_group'] = '21-30'
    userdf.loc[((userdf['age'] > 30) & (userdf['age'] <= 40)), 'age_group'] = '31-40'
    userdf.loc[((userdf['age'] > 40) & (userdf['age'] <= 50)), 'age_group'] = '41-50'
    userdf.loc[((userdf['age'] > 50) & (userdf['age'] <= 60)), 'age_group'] = '51-60'
    
    age_dummy = pd.get_dummies(userdf['age_group'])
    gender_dummy = pd.get_dummies(userdf['gender'])
    occupation_dummy = pd.get_dummies(userdf['occupation'])
    
    userdf = pd.concat([userdf, age_dummy, gender_dummy, occupation_dummy], axis = 1)
    
    
    cols = ['<=20', '21-30', '31-40', '41-50', '51-60', 'F', 'M',
       'administrator', 'artist', 'doctor', 'educator', 'engineer',
       'entertainment', 'executive', 'healthcare', 'homemaker', 'lawyer',
       'librarian', 'marketing', 'none', 'other', 'programmer', 'retired',
       'salesman', 'scientist', 'student', 'technician', 'writer']

    data = userdf[cols]
    #elbow(data) #best k=10
    #DBScan_epsilon(data)
    
    cols.append('userId')
    cluster_map = pd.DataFrame()
    cluster_map['userId'] = userdf[cols].userId.values
       
    if iskmeans == True:
        
        clustering_model = KMeans(n_clusters=10, random_state=1)
        clustering_model.fit(data)
        cluster_map['cluster'] = clustering_model.labels_

    else:
        clustering_model = DBSCAN(eps=1.4, min_samples=6)
        clustering_model.fit(data)
        cluster_map['cluster'] = clustering_model.labels_+1
        
    return cluster_map    
    
def avgRatingsPerCluster(ratings, movies,iskmeans):
    
    movieids = movies['movieId']
    userAvgCluster = {}
    
    cluster_map = clusterUsers(iskmeans)
    if iskmeans == True:     
        num_clusters = 10
    else:
        num_clusters = 33
        
    
    for i in range(0,num_clusters):
        
        cluster_df = cluster_map[cluster_map.cluster == i]
        userids = cluster_df['userId']
            
        for movieid in movieids: 
            
            ratingsofUsers = list(ratings[(ratings['movieId'] == movieid) & (ratings['userId'].isin(userids))]['rating'])
            
            if len(ratingsofUsers) == 0:
                avgusers = 0
            else:    
                avgusers = round(sum(ratingsofUsers) / len(ratingsofUsers),3)
                
            
            for userid in userids:
                userAvgCluster[userid,movieid] = avgusers
                    
    return userAvgCluster


def usersWithAtleastTenRatings():
    
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    users = pd.read_csv('../data/users.csv', sep=",")
    
    ratingscount = ratings.groupby(["userId"])[['rating']].count()
    ratingscountTen = ratingscount[ratingscount['rating'] >= 10] 
    print(ratingscountTen)
    users = ratingscountTen.reset_index()['userId']
    print(users)
    

def BernoulliGenerator():

    p = 0.25 #2,5,10,25,50,100% of productions enforce preferences
    generate = bernoulli.rvs(p, size=2077)
    listg = list(generate)
    df = DataFrame (listg,columns=['column'])
    #print(listg)
    df.to_csv('../data/bernoulli.csv', sep="," , index = False)


if __name__ == "__main__":     
    
    iskmeans = True
    #clusterUsers(iskmeans)
    
    """
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    movies = pd.read_csv('../data/movies.csv', sep=",")
    userAvgCluster = avgRatingsPerCluster(ratings, movies,iskmeans)
    print(len(userAvgCluster))
    """
