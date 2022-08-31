#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:36:16 2020

@author: tahereh
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import csr_matrix
import surprise
from surprise import KNNWithMeans, KNNBasic
from surprise import Dataset
from surprise import Reader, Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate

def predict_ratings_surprise_svd(ratings):
    ratings_dict = {'itemID': list(ratings.movieId),'userID': list(ratings.userId),'rating': list(ratings.rating)}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=.20)

    algo = SVD()
    
    algo.fit(trainset)
    predictions = algo.test(testset)

    #compute RMSE
    accuracy.rmse(predictions,verbose=True)
    
    
def predict_ratings_svd_RMSE(ratings_df):
    
    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0) 
    R = R_df.values
    R_sparse=csr_matrix(R)

    trainset_data, testset_data = train_test_split(R_sparse,test_size=0.20, random_state = 200)
    
    user_ratings_mean = np.mean(trainset_data, axis = 1)
    R_demeaned = trainset_data - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k = 30) 
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    print ('matrix-factorization SVD RMSE: %.2f' %rmse(all_user_predicted_ratings, testset_data))

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def predict_ratings_svd(ratings_df):
    
    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    R = R_df.to_numpy()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    
    U, sigma, Vt = svds(R_demeaned, k = 50) 
    sigma = np.diag(sigma)
    
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
    return preds_df

def MFrecommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations):

    user_row_number = userID - 1 # in ratings UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    sorted_user_predictions = sorted_user_predictions[sorted_user_predictions > 0] #only positive ratings
    
    """
    with open('../data/sorted_user_predictions.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(sorted_user_predictions.index.values,sorted_user_predictions.tolist()))
        #sorted_user_predictions.to_csv('../data/sorted_user_predictions.csv',sep=',', index=False)
    """    

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).iloc[:len(movies_df), :-1])
        
    return recommendations['movieId'][:num_recommendations].tolist() , recommendations['movieId'].tolist()



def knn(user_based):
    
    data = Dataset.load_builtin('ml-100k')
    sim_options = {
    'k':60,        
    "name": "cosine",
    "user_based": user_based,  # Compute  similarities between items
    }
    
    #algo = KNNBasic(sim_options=sim_options)
    algo = KNNWithMeans(sim_options=sim_options) # default for max_k = 40 and min_k = 1
    trainingSet = data.build_full_trainset()
    model = algo.fit(trainingSet)
    return model


def knnEval(user_based)  :
    
    #data = Dataset.load_builtin('ml-100k')
    data = Dataset.load_builtin('ml-1m')
    sim_options = {
    'k': [15, 20, 25, 30, 40, 50, 60,70,80,90,100]
    } # "name": "cosine", "user_based": user_based,  # Compute  similarities between items
    
    #trainset, testset = train_test_split(data, test_size=.20)
    #knnmeans_gs = KNNWithMeans(sim_options=sim_options)
    #predictions = algo.test(testset)
    #accuracy.rmse(predictions,verbose=True)
    
    knnmeans_gs = GridSearchCV(KNNWithMeans, sim_options, measures=['rmse', 'mae'], cv=5, n_jobs=5)
    knnmeans_gs.fit(data)
    
    y3 = knnmeans_gs.cv_results['mean_test_rmse']
    y4 = knnmeans_gs.cv_results['mean_test_mae']
    print('RMSE:', y3, 'MAE:',y4)
    
    
if __name__ == "__main__":    
    
    ratings_df = pd.read_csv('../../data/MovieLens/ml-1m/ratings_original.csv', sep=",")
    predictions_df = predict_ratings_svd(ratings_df)
    predictions_df.to_csv('../../data/SVD_ratings.csv', sep=',', index=False)
   