#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:31:27 2020

@author: tahereh
"""

import pandas as pd
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn import metrics
import csv
import random
import math
import os


def retriveUserBusinessVisits(movies, users, ratings, productions, userids, movieids, userVisits, isStatic):
    
    if isStatic == True:
        userids = users['userId']
        
    for userid in userids:
       
        ratings_per_user = ratings[(ratings['userId'] == userid)]  
    
        if isStatic == True:
            movies_per_user = pd.merge(ratings_per_user , movies , on=['movieId'], how='inner')
        else: #dynamic
            
            available_movies = movies[movies['movieId'].isin(movieids)]
            movies_per_user = pd.merge(ratings_per_user , available_movies , on=['movieId'], how='inner')
            
        productions_per_user = movies_per_user['productionId_x'].unique()
        
        if len(productions_per_user) != 0:
            if userid in userVisits: #static
                # update the previous productions_per_user
                previous_productions_per_user = userVisits[userid]
                previous_productions_per_user = list(previous_productions_per_user)
                previous_productions_per_user.extend(productions_per_user)
                    
                userVisits[userid] = list(set(previous_productions_per_user))
                
            else:
                 # create a new record
                userVisits[userid] = list(productions_per_user)
     
    return userVisits  

def getUserVisitRecord(userid, movieid,movies,userVisits):

    user_visit_bool = False
    production = movies[movies['movieId'] == movieid]['productionId_x'].tolist()
    productionid = production[0]
    if userid in userVisits.keys():
        visited_productions = userVisits[userid] 
        if productionid in visited_productions:
            user_visit_bool = True

    
    user_visit = []
    if user_visit_bool == True:
        user_visit = [0,1] #repeat
    else:
        user_visit = [1,0] #firstcomer
    
    return user_visit



def featureEngineering(moviedf,userdf,productiondf):
    
    #create dummies
    #movie features : genres, year, duration, country, language, duration, avg imdb vote or avg movielens rate
    if moviedf is not None:
        
        genres_dummy = moviedf['genres'].str.get_dummies(sep=', ') 
        moviedf.loc[((moviedf['year'] >= 1912) & (moviedf['year'] < 1970)), 'year_preference'] = 'classics'
        moviedf.loc[((moviedf['year'] >= 1970) & (moviedf['year'] < 2000)), 'year_preference'] = '70s'
        moviedf.loc[((moviedf['year'] >= 2000)), 'year_preference'] = 'new'
        year_dummy = pd.get_dummies(moviedf['year_preference'])
        moviedf.loc[((moviedf['duration'] >= 45) & (moviedf['duration'] < 60)), 'duration_preference'] = 'short'
        moviedf.loc[((moviedf['duration'] >= 60) & (moviedf['duration'] < 100)), 'duration_preference'] = 'medium'
        moviedf.loc[((moviedf['duration'] >= 100)), 'duration_preference'] = 'long'
        duration_dummy = pd.get_dummies(moviedf['duration_preference'])
        moviedf.loc[((moviedf['avg_vote'] >= 0) & (moviedf['avg_vote'] < 5)), 'vote_preference'] = 'low'
        moviedf.loc[((moviedf['avg_vote'] >= 5) & (moviedf['avg_vote'] < 7)), 'vote_preference'] = 'med'
        moviedf.loc[((moviedf['avg_vote'] >= 7)), 'vote_preference'] = 'high'
        vote_dummy = pd.get_dummies(moviedf['vote_preference'])
        moviedf['lang'] = np.where(moviedf.language.str.contains('English'),'English','non-English')
        language_dummy = moviedf['lang'].str.get_dummies(sep=', ') 
        
        moviedf = pd.concat([moviedf, genres_dummy, year_dummy, duration_dummy, vote_dummy, language_dummy], axis = 1)
        
    #-----------------------------------
    if userdf is not None:
        
        occupations = pd.read_csv('../data/occupations.csv', sep=",")
        userdf = pd.merge(userdf , occupations , on=['occupationId'], how='inner')
        userdf.loc[((userdf['age'] <= 20)), 'age_group'] = '<=20'
        userdf.loc[((userdf['age'] > 20) & (userdf['age'] <= 30)), 'age_group'] = '21-30'
        userdf.loc[((userdf['age'] > 30) & (userdf['age'] <= 40)), 'age_group'] = '31-40'
        userdf.loc[((userdf['age'] > 40) & (userdf['age'] <= 50)), 'age_group'] = '41-50'
        userdf.loc[((userdf['age'] > 50) & (userdf['age'] <= 60)), 'age_group'] = '51-60'
        age_dummy = pd.get_dummies(userdf['age_group'])
        occupation_dummy = pd.get_dummies(userdf['occupation'])
        userdf = pd.concat([userdf, age_dummy, occupation_dummy], axis = 1)
        #-------------------- 
        production_age_dummy = pd.get_dummies(productiondf['age_group'])
        production_occupation_dummy = productiondf['occupation'].str.get_dummies(sep=',')
        production_visit_dummy = pd.get_dummies(productiondf['visit'])
        productiondf = pd.concat([productiondf, production_age_dummy, production_occupation_dummy, production_visit_dummy], axis = 1)
    #-----------------------------------
    return moviedf, userdf, productiondf


def userVectorsRegression(moviedf, columns, users, ratings):
    
    user_vectors = {}
    userids = users['userId'].tolist()
    
    for userid in userids:
        ratings_per_user = ratings[(ratings['userId'] == userid)]  #& (ratings['rating'] >= 4)
        movies_per_user = pd.merge(ratings_per_user , moviedf , on=['movieId'], how='inner')
        df_columns = movies_per_user.columns[34:].tolist()  #columns after merging 
        
        movies_per_user['rating_class'] = np.where((movies_per_user['rating'] >= 4),1,0) 
        y = movies_per_user['rating_class']
        X = movies_per_user[df_columns]  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        weights = model.coef_
        weight_list = []
        for i,v in enumerate(weights):
            weight_list.append(round(v,3))
        
        user_vectors[userid] = weight_list

    return  user_vectors
        
   
def userVectorsAverage(movies, columns, users, ratings, userids, isStatic):
    
    user_vectors = {}
    
    if isStatic == True:
        userids = users['userId']
    
    for userid in userids:
        
        high_rated_movies = ratings[(ratings['userId'] == userid) & (ratings['rating'] >= 4)]['movieId']
        high_rated_movies_df = movies[movies['movieId'].isin(high_rated_movies)]
        user_vector = meanCols(high_rated_movies_df,columns)
        user_vectors[userid] = user_vector

    return user_vectors


def meanCols(df, columns):
    
    if len(df) != 0:
        mean_cols = df[columns].mean()#.sort_values(ascending=False)
        list_mean_cols = list(round(mean_cols,3))
        
    else:
         list_mean_cols = [0] * len(columns)
         
    return list_mean_cols
    
def UserMovieRelevanceMatrix(movies, users, ratings, userids, movieids, isStatic):
    
    moviedf, userdf , productiondf = featureEngineering(movies,None, None)  
    columns = moviedf.columns[33:].tolist()
    columns2 = columns.copy()
    columns2.append('movieId')
    movie_vectors = moviedf[columns2]
    
    if isStatic == True:
        movieids = movie_vectors['movieId']
    
   
    UserMovieMatrix = {}
    for userid in user_vectors.keys():
        for movieid in movieids:
            
            user_vector = user_vectors[userid]
            movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][columns].values.tolist()
            
            if all(v == 0 for v in user_vector):
                cos_sim = 0
            else:
                cos_sim = 1 - spatial.distance.cosine(user_vector,movie_vector)
              
            UserMovieMatrix[userid, movieid] = round(cos_sim,3)

    return UserMovieMatrix
    

def productionVectors(columns, movies, ratings, productions):   
    
    columns2 = columns.copy()
    columns2.append('movieId')
    columns2.append('productionId')
    movies_merged = pd.merge(productions , movies , on=['production_company'], how='inner')
    production_vectors = movies_merged[columns2]
    return production_vectors

    
def movieVectors(columns, movies, ratings, productions):   
    
    columns2 = columns.copy()
    columns2.append('movieId')
    movies_merged = pd.merge(productions , movies , on=['production_company'], how='inner')
    movie_vectors = movies_merged[columns2]
    return movie_vectors

def MovieUserRelevanceMatrix(movies, users, ratings, userids, movieids, userVisits, isStatic):
    
    productions = pd.read_csv('../data/productions.csv', sep=",")
    moviedf, userdf, productiondf = featureEngineering(None,users,productions)  
    user_columns = userdf.columns[7:].tolist() 
    pro_columns = productiondf.columns[5:].tolist() 

    #read ranodm preferences for businesses
    #equate the scores to each business' movies => movie prefernce vector
    movie_vectors = movieVectors(pro_columns, movies, ratings, productiondf) 

    columns2 = user_columns.copy()
    columns2.append('userId')
    user_vectors = userdf[columns2]
    
    if isStatic == True:
        userids = user_vectors['userId']
        movieids = movie_vectors['movieId']
    
    MovieUserMatrix = {}
    userVisits = retriveUserBusinessVisits(movies, users, ratings,productions, userids, movieids, userVisits, isStatic)
    
    for movieid in movieids:
        
        age_care = movie_vectors[movie_vectors['movieId'] == movieid]['age_care']
        occupation_care = movie_vectors[movie_vectors['movieId'] == movieid]['occupation_care']
        visit_care = movie_vectors[movie_vectors['movieId'] == movieid]['visit_care']
        
        for userid in userids:
            
            pro_columns = productiondf.columns[8:].tolist() 
            user_columns = userdf.columns[7:].tolist()
            #----------------
            if (age_care.any() == 0) & (occupation_care.any() == 0) & (visit_care.any() == 0):
                pro_columns = []
                user_columns = []
             
                
            elif (age_care.any() == 1) & (occupation_care.any() == 1) & (visit_care.any() == 1):
                
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist()
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist()
                
                
            elif (age_care.any() == 1) & (occupation_care.any() == 1) & (visit_care.any() == 0):
               
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist() 
                   
                del movie_vector[0][26:28]
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist() 
                
            
                
            elif (age_care.any() == 1) & (occupation_care.any() == 0) & (visit_care.any() == 1):
                
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist()
                   
                del movie_vector[0][5:26]
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist() 

            elif (age_care.any() == 1) & (occupation_care.any() == 0) & (visit_care.any() == 0):
                
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist()

                del movie_vector[0][5:28]
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist() 
                  
                
            
            elif (age_care.any() == 0) & (occupation_care.any() == 1) & (visit_care.any() == 1):
                
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist()

                del movie_vector[0][0:5]
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist() 
                
                
                
            elif (age_care.any() == 0) & (occupation_care.any() == 1) & (visit_care.any() == 0):
               
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist()

                del movie_vector[0][0:5]
                del movie_vector[0][26:28]
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist() 
                
                
            elif (age_care.any() == 0) & (occupation_care.any() == 0) & (visit_care.any() == 1):
               
                movie_vector = movie_vectors[movie_vectors['movieId'] == movieid][pro_columns].values.tolist()

                del movie_vector[0][0:26]
                user_vector = user_vectors[user_vectors['userId'] == userid][user_columns].values.tolist() 
  
            #----------------
            if (pro_columns == []) & (user_columns == []):
                cos_sim = 1
            else:   
                #----------------
                uservisit = getUserVisitRecord(userid, movieid, movies, userVisits)
                for i in user_vector:
                    i.extend(uservisit) 
                if (age_care.any() == 1) & (occupation_care.any() == 1) & (visit_care.any() == 0):
                     del user_vector[0][26:28]
                elif (age_care.any() == 1) & (occupation_care.any() == 0) & (visit_care.any() == 1):
                     del user_vector[0][5:26]    
                elif (age_care.any() == 1) & (occupation_care.any() == 0) & (visit_care.any() == 0):
                     del user_vector[0][5:28] 
                elif (age_care.any() == 0) & (occupation_care.any() == 1) & (visit_care.any() == 1):
                     del user_vector[0][0:5]
                elif (age_care.any() == 0) & (occupation_care.any() == 1) & (visit_care.any() == 0): 
                    del user_vector[0][0:5]
                    del user_vector[0][26:28]
                elif (age_care.any() == 0) & (occupation_care.any() == 0) & (visit_care.any() == 1):
                     del user_vector[0][0:26]
                #----------------
                cos_sim = 1 - spatial.distance.cosine(movie_vector, user_vector)
            MovieUserMatrix[userid,movieid] = round(cos_sim,3)
        
    return MovieUserMatrix, userVisits

   

def MutualRelevanceMatrix(userids, movieids, userVisits, isStatic):

    movies = pd.read_csv('../data/movies.csv', sep=",")
    users = pd.read_csv('../data/users.csv', sep=",")
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    
    if isStatic == True:
        movieids = movies['movieId']
        userids = users['userId']
        
    UserMovieMatrix = UserMovieRelevanceMatrix(movies, users, ratings, userids, movieids, isStatic)
    print('UserMovieMatrix is created!')
    MovieUserMatrix, userVisits = MovieUserRelevanceMatrix(movies, users, ratings, userids, movieids, userVisits, isStatic)
    print('MovieUserMatrix is created!')
    #----------------------------------------
    
    with open('../data/MovieLensProductionUserMatrix_5%.csv', 'w') as csv_file:  
            writer = csv.writer(csv_file)
            writer.writerow(['productionId', 'userId', 'similarity'])
            
            for userid in userids:
                for movieid in movieids:
                    productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]
                    rel_mu = MovieUserMatrix[userid,movieid]
                    writer.writerow([productionid,userid,rel_mu])

    print('MovieLensProductionUserMatrix is created!!!')
    
    #----------------------------------------
    
    if isStatic == False:
        for userid in userids:
            for movieid in movieids:
                rel_um = UserMovieMatrix[userid,movieid]  
                rel_mu = MovieUserMatrix[userid,movieid]
                mutual_rel = round(rel_um * rel_mu ,3)
                return mutual_rel
    
    else:
        with open('../data/MovieLensRelevanceMatrix_5%.csv', 'w') as csv_file:  
            writer = csv.writer(csv_file)
            writer.writerow(['userId', 'movieId', 'relevance'])
            
            for userid in userids:
                for movieid in movieids:
                    
                    rel_um = UserMovieMatrix[userid,movieid]  
                    rel_mu = MovieUserMatrix[userid,movieid]  
                    mutual_rel = round(rel_um * rel_mu ,3) 
                    writer.writerow([userid,movieid,mutual_rel])
                
          
   
if __name__ == "__main__":     
    
   userVisits = {} 
   MutualRelevanceMatrix(None, None, userVisits, True)
   
    
    
    
    
    
    
    
    
    
    