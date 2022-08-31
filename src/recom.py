#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:22:46 2020

@author: tahereh
"""


import math
import random
import pandas as pd
import numpy as np
import operator
from datetime import timedelta
from datetime import datetime 
import os
import csv
import time
from relevanceMatrix import MutualRelevanceMatrix , featureEngineering , userVectorsAverage, productionVectors, retriveUserBusinessVisits
from userFeedback import getUserAcceptedMovie
from SVD import MFrecommend_movies, predict_ratings_svd, knn
from evaluation import avgBusinessSatisfaction, avgUserSatisfaction , avgItemListsSimilarity, DCG
from rankggtest import footrule_aggregation , borda_aggregation , local_kemenization, highest_rank
from rankaggregator import RankAggregator
from dataPreprocessing import avgRatingsPerCluster
userVisits = {}  #records the visits of users for companies

def nextTime(rateParameter): 
    return (-math.log(1.0 - random.random())) / (rateParameter)

def moviesTime(poisson_rating, movies ,total_number_of_movies):
    
    column_names = ['poisson time', 'timestamp',  'movieId']
    df = pd.DataFrame(columns = column_names)
    df['timestamp']= pd.to_datetime(df['timestamp'])
    df['movieId'] = df['movieId'].astype(int)
    
    movie_time = 0
    movie_ids = movies['movieId']
    
    
    for i in range(1,total_number_of_movies+1): 
    
        movie_nexttime = nextTime(poisson_rating)
        movie_time = movie_time + movie_nexttime
        movie_time = round(movie_time,2)
        df.loc[i, 'poisson time'] = movie_time
        #-----------------
        timestamp = datetime(year=2020, day=20, month=4, hour=7, minute=0) #start time 
        timestamp1 = timestamp + timedelta(minutes=movie_time)
        timestamp = timestamp1 - timedelta(microseconds=timestamp1.microsecond)
        df.loc[i, 'timestamp'] = timestamp
        #-----------------
        random_movie = np.random.choice(movie_ids, 1, replace=False)
        random_movie_int = int(random_movie[0])
        remaining_volume = movies[(movies['movieId'] == random_movie_int)]['remaining_volume'].iloc[0]
        
        if remaining_volume > 1:
            df.loc[i, 'movieId'] = random_movie_int
            movies.loc[(movies['movieId'] == random_movie_int), 'remaining_volume'] = remaining_volume - 1
            
        elif remaining_volume == 1:
             df.loc[i, 'movieId'] = random_movie_int
             movie_ids = list(set(movie_ids) - set(random_movie))
         
    df.to_csv("../data/movies_times.csv", sep=',',index=False,date_format='%Y-%m-%d %H:%M:%S' ) 
    return df

def RRS(random_user, available_movieids, sim_dict, isStatic, number_of_offers, userVisits):
    # recommend movies with highest scores based on the mutual similarity dictionary
    unique_available_movieids = list(set(available_movieids))
    similarities = {}
    
    for movieid in unique_available_movieids:
        
        if isStatic == True:
            similarities[movieid] = sim_dict[random_user, movieid]
            
        else:
            similarities[movieid] = MutualRelevanceMatrix([random_user], [movieid], userVisits, isStatic)
    
    sorted_d = sorted(similarities.items(), key=operator.itemgetter(1), reverse = True)
    sorted_movies = [i[0] for i in sorted_d]
    sorted_scores = [i[1] for i in sorted_d]
    return sorted_movies[0:number_of_offers] , sorted_scores[0:number_of_offers] 


def Random(availabe_movies, number_of_offers):
    
   if len(availabe_movies) < number_of_offers:
       random_movieids = random.sample(list(availabe_movies),len(availabe_movies)) 
           
   else:
       random_movieids = random.sample(list(availabe_movies),number_of_offers) 
    
   return  random_movieids
    

def rateBased(recom_list, ratings, userid, number_of_offers, userAvgCluster):
    
    ratings_dict = {}
    if len(recom_list) != 0:
    
        for movieid in recom_list:
        
            movie_rating = ratings[(ratings['userId'] == userid) & (ratings['movieId'] == movieid)]['rating']
            
            if movie_rating.empty == False:
                ratings_dict[movieid] = movie_rating.tail(1).iloc[0]
            
            else: # if user does not have rating for a movie then compute the average ratings of all users for that movie
                
                ratingsAllusers = list(ratings[ratings['movieId'] == movieid]['rating'])
                if len(ratingsAllusers) == 0:
                    avgAllusers = 0
                else:    
                    avgAllusers = sum(ratingsAllusers) / len(ratingsAllusers) 
                
                ratings_dict[movieid] = avgAllusers
                
                """
                #userAvgCluster = avgRatingsPerCluster(ratings, movieid)
                avgusers = userAvgCluster[userid,movieid]
                ratings_dict[movieid] = avgusers
                """
                
        sorted_d = sorted(ratings_dict.items(), key=operator.itemgetter(1), reverse = True)
        sorted_movies = [i[0] for i in sorted_d]
        sorted_scores = [i[1] for i in sorted_d]
        
        return sorted_movies[0:number_of_offers] , sorted_scores[0:number_of_offers], sorted_movies
    
    

def BusinessOriented(userid , recom_list, movies, number_of_offers, sim_dict):

    re_ranked_dict = {}
          
    for movieid in recom_list:
        
        productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]
        similarity = sim_dict[(productionid,userid)] 
        re_ranked_dict[movieid] = similarity
    
    sorted_d = sorted(re_ranked_dict.items(), key=operator.itemgetter(1), reverse = True)
    sorted_movies = [i[0] for i in sorted_d]
    sorted_scores = [i[1] for i in sorted_d]
    
    return sorted_movies[0:number_of_offers], sorted_scores[0:number_of_offers], sorted_movies


def rankAggregation(rate_based_list, business_preference_based_list, availabe_movies, number_of_offers):
    
    agg = RankAggregator()
    
    rank_list = []
    rank_list.append(rate_based_list)
    rank_list.append(business_preference_based_list)
    sorted_b = agg.borda(rank_list) #borda , #dowdall #average_rank 
    sorted_movies = [i[0] for i in sorted_b]
    return sorted_movies[0:number_of_offers]


def CFrecommend_movies(knnModel, random_user, movies, number_of_offers):
    
    knnPredictions = {}
    
    for movieid in movies: 
        
        prediction = knnModel.predict(str(random_user), str(movieid))
        knnPredictions[movieid] = prediction.est
        
    sorted_d = sorted(knnPredictions.items(), key=operator.itemgetter(1), reverse = True)
    sorted_movies = [i[0] for i in sorted_d]
    sorted_scores = [i[1] for i in sorted_d]
    return sorted_movies[0:number_of_offers] , sorted_scores[0:number_of_offers], sorted_movies

def split_ratings(data, test_ratio=0.3):

    #date: ratings 
    train_file = "../data/ratings_train.csv"
    test_file = "../data/ratings_test.csv"
    
    mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
    neg_mask = [not x for x in mask]
    train_data, test_data = data[mask], data[neg_mask]
    train_data.to_csv(train_file, sep=',', index=False, date_format='%Y-%m-%d %H:%M:%S') 
    test_data.to_csv(test_file, sep=',', index=False, date_format='%Y-%m-%d %H:%M:%S')

    
def main(knnModel_item_based, userAvgCluster,newMetrics, userPre_vectors, movie_vectors, Mcolumns, production_vectors, userAtt_vectors, userVisitStatus, user_columns, pro_columns):
    
    start_time = time.time()
    #----------------
    num_generated_records = 50
    users_poisson_rating = 1 #1-5
    movies_poisson_rating = 5 #1-5, flash crowd
    number_of_offers = 5
    isStatic = True        # if user and movie data is static or dynamic
    isDeterministic = True # if user selects deterministically or probabilistically
    isMoviesFlashCrowd = False #if all movies arrive at the same time
    #----------------
    users_file = "../data/users.csv"
    movies_file = "../data/movies.csv"
    recom_file = "../data/recom.csv"
    ratings_file = "../data/ratings.csv"
    users = pd.read_csv(users_file, sep=",")
    movies = pd.read_csv(movies_file, sep=",")
    ratings = pd.read_csv(ratings_file, sep=",")
    
    total_number_of_movies =  sum(movies['volume']) 
    num_users = len(users)
    user_time = 0
    user_ids = users['userId']
    sim_dict = {}
    MovieUser_sim_dict = {}
    #----------------
    recom_column_names = ['poisson time', 'timestamp', 'userid', 'available_movieids', 
                    'RRS_recom', 'RRS_scores'  , 'accepted_movie', 'Random_recom', 
                    'CF_item_based_recom', 'CF_item_based_ratings' 
                    ,'BusinessOriented_recom', 'BusinessOriented_scores', 'RankAggregation_recom']
    
    recom_df = pd.DataFrame(columns = recom_column_names)
    recom_df['timestamp']= pd.to_datetime(recom_df['timestamp'])
    recom_df['RRS_recom'] = recom_df['RRS_recom'].astype(int)
    #----------------
    availabe_movies = []
    if isMoviesFlashCrowd == False:
        movies_times = moviesTime(movies_poisson_rating, movies, total_number_of_movies) 
    else:

        movieids = movies['movieId']
        for movieid in movieids:
            coupvolume = movies[movies['movieId'] == movieid]['volume'].iloc[0]
            availabe_movies.append(coupvolume * [movieid])    
        
        availabe_movies = [val for sublist in availabe_movies for val in sublist]    
        
        
    if isStatic == True:
        rel_matrix = pd.read_csv('../data/MovieLensRelevanceMatrix.csv', sep=",")
        for row in rel_matrix.to_dict('records'):
              key = (row['userId'], row['movieId'])
              sim_dict[key] = row['relevance']
    
    
    MovieUserMatrix = pd.read_csv('../data/MovieLensProductionUserMatrix.csv', sep=",")
    for row in MovieUserMatrix.to_dict('records'):
          key = (row['productionId'], row['userId'])
          MovieUser_sim_dict[key] = row['similarity']    
    #------------------------------------------------------------- 
    for i in range(1,num_generated_records+1):
        
        #-----------------
        user_nexttime = nextTime(users_poisson_rating)
        user_time = user_time + user_nexttime
        user_time = round(user_time,2)
        recom_df.loc[i, 'poisson time'] = user_time
    
        timestamp = datetime(year=2020, day=20, month=4, hour=7, minute=0, second=0) #start time 
        timestamp1 = timestamp + timedelta(minutes=user_time)
        timestamp = timestamp1 - timedelta(microseconds=timestamp1.microsecond)
        recom_df.loc[i, 'timestamp'] = timestamp
        #-----------------
        random_user = np.random.choice(user_ids, 1, replace=False)
        
        if num_generated_records <= num_users: 
            user_ids = list(set(user_ids) - set(random_user)) ##needed if we want without replacement
        
        recom_df.loc[i, 'userid'] = random_user[0]
        random_user = int(random_user)
        #----------------------
        if isMoviesFlashCrowd == False:
            availabe_movies = movies_times[movies_times['poisson time'] <= user_time]['movieId']  
            availabe_movies = [int(obj) for obj in availabe_movies]
            available_movieids_str = [str(integer) for integer in availabe_movies]
            available_movieids = ", ".join(available_movieids_str)
            recom_df.loc[i, 'available_movieids'] = available_movieids
        else:
            movie_strings = [str(integer) for integer in movieids]
            available_movieids = ", ".join(movie_strings)
            recom_df.loc[i, 'available_movieids'] = available_movieids
        #----------------------
        if isMoviesFlashCrowd == False:
            sorted_offered_movies , sorted_scores = RRS(random_user, availabe_movies, sim_dict, isStatic, number_of_offers, userVisits)
        else:
            sorted_offered_movies , sorted_scores = RRS(random_user, movieids, sim_dict, isStatic, number_of_offers, userVisits)
        
        sorted_movie_strings = [str(integer) for integer in sorted_offered_movies]
        RRS_offered_movieids = ", ".join(sorted_movie_strings)
        recom_df.loc[i, 'RRS_recom'] = RRS_offered_movieids

        sorted_scores_strings = [str(round(integer,2)) for integer in sorted_scores]
        sorted_scores_offered_movieids = ", ".join(sorted_scores_strings)
        recom_df.loc[i, 'RRS_scores'] = sorted_scores_offered_movieids
        #----------------------
        
        if len(availabe_movies)!= 0:
            #----------------------------
            #Random
            if isMoviesFlashCrowd == False:
               sorted_random_movies   = Random(availabe_movies, number_of_offers)  
            else:
                sorted_random_movies  = Random(movieids, number_of_offers)
            
            sorted_random_movies_strings = [str(integer) for integer in sorted_random_movies]
            random_offered_movieids = ", ".join(sorted_random_movies_strings)
            recom_df.loc[i, 'Random_recom'] = random_offered_movieids
            
            #----------------------------
            if isMoviesFlashCrowd == False:
                sorted_CFI_movies , sorted_CFI_ratings , all_sorted_CFI_movies = CFrecommend_movies(knnModel_item_based, random_user, availabe_movies, number_of_offers)
            else:   
                sorted_CFI_movies , sorted_CFI_ratings , all_sorted_CFI_movies = CFrecommend_movies(knnModel_item_based, random_user, movieids, number_of_offers)
            
            
            sorted_CFI_movies_strings = [str(integer) for integer in sorted_CFI_movies]
            rated_offered_movieids = ", ".join(sorted_CFI_movies_strings)
            recom_df.loc[i, 'CF_item_based_recom'] = rated_offered_movieids

            sorted_ratings_str = [str(round(integer,2)) for integer in sorted_CFI_ratings]
            sorted_ratings_scores = ", ".join(sorted_ratings_str)
            recom_df.loc[i, 'CF_item_based_ratings'] = sorted_ratings_scores
            
            #----------------------
            #----------------------
            if isMoviesFlashCrowd == False:
                business_pref_movieids,business_pref_scores, all_sorted_buOri_movies = BusinessOriented(random_user, availabe_movies, movies, number_of_offers, MovieUser_sim_dict) 
            else:
                business_pref_movieids,business_pref_scores, all_sorted_buOri_movies = BusinessOriented(random_user, movieids, movies, number_of_offers, MovieUser_sim_dict) 
                
            business_pref_movieids_str = [str(integer).replace(".0", '') for integer in business_pref_movieids]
            business_pref_movieids_offer = ", ".join(business_pref_movieids_str)
            recom_df.loc[i, 'BusinessOriented_recom'] = business_pref_movieids_offer
            
            sorted_business_pref_str = [str(round(integer,2)) for integer in business_pref_scores]
            sorted_business_pref_scores = ", ".join(sorted_business_pref_str)
            recom_df.loc[i, 'BusinessOriented_scores'] = sorted_business_pref_scores
            #----------------------
            # rank aggregation
            rankagg_movieids = rankAggregation(all_sorted_CFI_movies , all_sorted_buOri_movies, availabe_movies, number_of_offers)
            rankagg_movieids_str = [str(integer).replace(".0", '') for integer in rankagg_movieids]
            rankagg_movieids_offered = ", ".join(rankagg_movieids_str)
            recom_df.loc[i, 'RankAggregation_recom'] = rankagg_movieids_offered
            #---------------------
        else:
             recom_df.loc[i, 'RRS_recom'] = ''
             recom_df.loc[i, 'Random_recom'] = ''
             recom_df.loc[i, 'CF_item_based_recom'] = ''
             recom_df.loc[i, 'BusinessOriented_recom'] = ''    
             recom_df.loc[i, 'RankAggregation_recom'] = ''
        #----------------------
        #----------------------
        # user accept an offer
        if len(RRS_offered_movieids) != 0:
            
            userAcceptedMovie = getUserAcceptedMovie(random_user, availabe_movies, isDeterministic, movies) #sorted_offered_movies
            recom_df.loc[i, 'accepted_movie'] = userAcceptedMovie 
            #------------------
            #### update user-visits ###
            accepted_movie_production = movies[movies['movieId'] == userAcceptedMovie]['productionId_x'].iloc[0]
            if (len(userVisits) != 0) & (random_user in userVisits.keys()): 
                visitlist = userVisits[random_user]
                if (accepted_movie_production not in visitlist):
                    visitlist.append(accepted_movie_production)
                    userVisits[random_user] = visitlist
            else:
                 visitlist = []
                 visitlist.append(accepted_movie_production)
                 userVisits[random_user] = visitlist
        #----------------------
            if isMoviesFlashCrowd == False:
                
                available_accepted_movies = movies_times[movies_times['movieId'] == userAcceptedMovie]
                if len(available_accepted_movies)!= 0:
                    first_occurance_time = available_accepted_movies['poisson time'].iloc[0]
                    movies_times = movies_times[(movies_times['poisson time'] != first_occurance_time)]
            else:
                availabe_movies.remove(userAcceptedMovie)
        
        #----------------------
    #---------------------- 
    recom_df.to_csv(recom_file, sep=',', index=False, date_format='%Y-%m-%d %H:%M:%S')  
    #----------------------
    user_based = False
    use_dcg = False
    
    if newMetrics == False:
        rrs_aus, random_aus, cfi_aus , bo_aus , rankagg_aus = avgUserSatisfaction(recom_df,ratings, knnModel_item_based , user_based, userAvgCluster, use_dcg)
        rrs_abs, random_abs, cfi_abs, bo_abs , rankagg_abs = avgBusinessSatisfaction(recom_df, movies, MovieUser_sim_dict)
       
    print("--- %s seconds ---" % (time.time() - start_time))
    #-------------------------------------------------------------      
    return rrs_aus , random_aus, cfi_aus , bo_aus , rankagg_aus , rrs_abs , random_abs, cfi_abs , bo_abs , rankagg_abs
    

def getUsersMoviesVectors(movies, users, ratings, userids, isStatic):
    
    moviedf, userdf , productiondf = featureEngineering(movies, None, None)  
    columns = moviedf.columns[33:].tolist()
    user_vectors = userVectorsAverage(moviedf, columns , users, ratings, userids, isStatic)
    columns2 = columns.copy()
    columns2.append('movieId')
    movie_vectors = moviedf[columns2]
    return user_vectors, movie_vectors, columns


def getProductionsUsersVectors(movies, users, ratings, productions, userVisits, isStatic):
    
    moviedf, userdf, productiondf = featureEngineering(None,users,productions)  
    user_columns = userdf.columns[7:].tolist() 
    pro_columns = productiondf.columns[5:].tolist() 
    production_vectors = productionVectors(pro_columns, movies, ratings, productiondf) 
    columns2 = user_columns.copy()
    columns2.append('userId')
    user_vectors = userdf[columns2]
    userids = user_vectors['userId']
    movieids = production_vectors['movieId']
    pro_columns = productiondf.columns[8:].tolist() 
    user_columns = userdf.columns[7:].tolist()
    userVisits = retriveUserBusinessVisits(movies, users, ratings,productions, userids, movieids, userVisits, isStatic)
    
    return production_vectors, user_vectors, userVisits, user_columns, pro_columns
    

if __name__ == "__main__":  
    
    newMetrics = False

    if  newMetrics == False:
        iskmeans = True
        knnModel_item_based = knn(False)
        ratings = pd.read_csv('../data/ratings.csv', sep=",")
        movies = pd.read_csv('../data/movies.csv', sep=",")
        userAvgCluster = avgRatingsPerCluster(ratings,movies,iskmeans)
        
        main(knnModel_item_based, userAvgCluster, newMetrics, None, None, None, None, None, None, None, None) 
        
    else:
        
        knnModel_item_based = knn(False)
        users = pd.read_csv('../data/users.csv', sep=",")
        ratings = pd.read_csv('../data/ratings.csv', sep=",")
        movies = pd.read_csv('../data/movies.csv', sep=",")
        productions = pd.read_csv('../data/productions.csv', sep=",")
        userids = users['userId']
        userVisitStatus = {} 
        
        
        userPre_vectors, movie_vectors, Mcolumns = getUsersMoviesVectors(movies, users, ratings, userids, True)
        production_vectors, userAtt_vectors, userVisitStatus, user_columns, pro_columns = getProductionsUsersVectors(movies, users, ratings, productions, userVisitStatus, True)
               
        main(knnModel_item_based, None, newMetrics, userPre_vectors, movie_vectors, Mcolumns, production_vectors, userAtt_vectors, userVisitStatus, user_columns, pro_columns) 
    
    
