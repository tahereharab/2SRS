#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:40:40 2020

@author: tahereh
"""

import numpy as np
import pandas as pd
import math
from dataPreprocessing import avgRatingsPerCluster



def avgBusinessSatisfaction(recom, movies, sim_dict):
    #------------------------
    rrs_usersPerproduction , random_usersPerproduction, cf_item_usersPerproduction , busiOri_usersPerproduction , rankagg_usersPerproduction = getListUsersforProduction(recom, movies) 
    #------------------------
    rrs_mean_satisfaction = 0
    random_mean_satisfaction = 0
    cf_item_mean_satisfaction = 0
    busiOri_mean_satisfaction = 0
    rankagg_mean_satisfaction = 0
    
    """
    sim_dict = {}
    MovieUserMatrix = pd.read_csv('../data/MovieLensProductionUserMatrix.csv', sep=",")
    for row in MovieUserMatrix.to_dict('records'):
          key = (row['productionId'], row['userId'])
          sim_dict[key] = row['similarity']
    """      
    #------------------------
    rrs_sum_scores = 0
    if len(rrs_usersPerproduction) != 0:
        for key in rrs_usersPerproduction.keys():
            
            user_list = rrs_usersPerproduction[key]
            rrs_production_sum_scores = 0
            
            for userid in user_list:
                
                similarity = sim_dict[(key,userid)]
                if similarity >= 0.8:
                    rrs_production_sum_scores = rrs_production_sum_scores + 5
                elif  (similarity >= 0.6) & (similarity < 0.8):
                    rrs_production_sum_scores = rrs_production_sum_scores + 4
                elif  (similarity >= 0.4) & (similarity < 0.6):
                    rrs_production_sum_scores = rrs_production_sum_scores + 3
                elif  (similarity >= 0.2) & (similarity < 0.4):
                    rrs_production_sum_scores = rrs_production_sum_scores + 2
                elif  (similarity >= 0.0) & (similarity < 0.2):
                    rrs_production_sum_scores = rrs_production_sum_scores + 1
                else: 
                    rrs_production_sum_scores = rrs_production_sum_scores + 0    
            
            rrs_production_mean = round(rrs_production_sum_scores/len(user_list),2) 
            rrs_sum_scores = rrs_sum_scores + rrs_production_mean
                
        rrs_mean_satisfaction = round(rrs_sum_scores/len(rrs_usersPerproduction),2)
        print( 'Average Business Satisfaction for RRS = ', rrs_mean_satisfaction)
    #------------------------
    random_sum_scores = 0
    if len(random_usersPerproduction) != 0:
        for key in random_usersPerproduction.keys():
            
            user_list = random_usersPerproduction[key]
            random_production_sum_scores = 0

            for userid in user_list:
                
                similarity = sim_dict[(key,userid)]
                if similarity >= 0.8:
                    random_production_sum_scores = random_production_sum_scores + 5
                elif  (similarity >= 0.6) & (similarity < 0.8):
                    random_production_sum_scores = random_production_sum_scores + 4
                elif  (similarity >= 0.4) & (similarity < 0.6):
                    random_production_sum_scores = random_production_sum_scores + 3
                elif  (similarity >= 0.2) & (similarity < 0.4):
                    random_production_sum_scores = random_production_sum_scores + 2
                elif  (similarity >= 0.0) & (similarity < 0.2):
                    random_production_sum_scores = random_production_sum_scores + 1
                else: 
                    random_production_sum_scores = random_production_sum_scores + 0    
            
            random_production_mean = round(random_production_sum_scores/len(user_list),2) 
            random_sum_scores = random_sum_scores + random_production_mean
                
        random_mean_satisfaction = round(random_sum_scores/len(random_usersPerproduction),2)
        print( 'Average Business Satisfaction for Random = ', random_mean_satisfaction)    
    #------------------------
    cf_item_sum_scores = 0
    for key in cf_item_usersPerproduction.keys():
        
        user_list = cf_item_usersPerproduction[key]
        cf_item_production_sum_scores = 0
        
        for userid in user_list:
            
            similarity = sim_dict[(key,userid)]
            if similarity >= 0.8:
                cf_item_production_sum_scores = cf_item_production_sum_scores + 5
            elif  (similarity >= 0.6) & (similarity < 0.8):
                cf_item_production_sum_scores = cf_item_production_sum_scores + 4
            elif  (similarity >= 0.4) & (similarity < 0.6):
                cf_item_production_sum_scores = cf_item_production_sum_scores + 3
            elif  (similarity >= 0.2) & (similarity < 0.4):
                cf_item_production_sum_scores = cf_item_production_sum_scores + 2
            elif  (similarity >= 0.0) & (similarity < 0.2):
                cf_item_production_sum_scores = cf_item_production_sum_scores + 1
            else: 
                cf_item_production_sum_scores = cf_item_production_sum_scores + 0
        
        cf_item_production_mean = round(cf_item_production_sum_scores/len(user_list),2) 
        cf_item_sum_scores = cf_item_sum_scores + cf_item_production_mean
            
    cf_item_mean_satisfaction = round(cf_item_sum_scores/len(cf_item_usersPerproduction),2)
    print( 'Average Business Satisfaction for item_based CF = ', cf_item_mean_satisfaction)
    #------------------------
    busiOri_sum_scores = 0
    if len(busiOri_usersPerproduction) != 0:
        for key in busiOri_usersPerproduction.keys():
            
            user_list = busiOri_usersPerproduction[key]
            busiOri_production_sum_scores = 0
            
            for userid in user_list:
                
                similarity = sim_dict[(key,userid)]
                if similarity >= 0.8:
                    busiOri_production_sum_scores = busiOri_production_sum_scores + 5
                elif  (similarity >= 0.6) & (similarity < 0.8):
                    busiOri_production_sum_scores = busiOri_production_sum_scores + 4
                elif  (similarity >= 0.4) & (similarity < 0.6):
                    busiOri_production_sum_scores = busiOri_production_sum_scores + 3
                elif  (similarity >= 0.2) & (similarity < 0.4):
                    busiOri_production_sum_scores = busiOri_production_sum_scores + 2
                elif  (similarity >= 0.0) & (similarity < 0.2):
                    busiOri_production_sum_scores = busiOri_production_sum_scores + 1
                else: 
                    busiOri_production_sum_scores = busiOri_production_sum_scores + 0    
            
            busiOri_production_mean = round(busiOri_production_sum_scores/len(user_list),2) 
            busiOri_sum_scores = busiOri_sum_scores + busiOri_production_mean
            #print(busiOri_production_mean)
                
        busiOri_mean_satisfaction = round(busiOri_sum_scores/len(busiOri_usersPerproduction),2)
        print( 'Average Business Satisfaction for Business-Oriented = ', busiOri_mean_satisfaction)
    #------------------------
    rankagg_sum_scores = 0
    if len(rankagg_usersPerproduction) != 0:
        for key in rankagg_usersPerproduction.keys():
            
            user_list = rankagg_usersPerproduction[key]
            rankagg_production_sum_scores = 0
            
            for userid in user_list:
                
                similarity = sim_dict[(key,userid)]
                if similarity >= 0.8:
                    rankagg_production_sum_scores = rankagg_production_sum_scores + 5
                elif  (similarity >= 0.6) & (similarity < 0.8):
                    rankagg_production_sum_scores = rankagg_production_sum_scores + 4
                elif  (similarity >= 0.4) & (similarity < 0.6):
                    rankagg_production_sum_scores = rankagg_production_sum_scores + 3
                elif  (similarity >= 0.2) & (similarity < 0.4):
                    rankagg_production_sum_scores = rankagg_production_sum_scores + 2
                elif  (similarity >= 0.0) & (similarity < 0.2):
                    rankagg_production_sum_scores = rankagg_production_sum_scores + 1
                else: 
                    rankagg_production_sum_scores = rankagg_production_sum_scores + 0    
            
            rankagg_production_mean = round(rankagg_production_sum_scores/len(user_list),2) 
            rankagg_sum_scores = rankagg_sum_scores + rankagg_production_mean
                
        rankagg_mean_satisfaction = round(rankagg_sum_scores/len(rankagg_usersPerproduction),2)
        print( 'Average Business Satisfaction for Rank Aggregation = ', rankagg_mean_satisfaction)
    #------------------------
    return rrs_mean_satisfaction  , random_mean_satisfaction, cf_item_mean_satisfaction , busiOri_mean_satisfaction, rankagg_mean_satisfaction
    
def getListUsersforProduction(recom, movies):
    
    recom.fillna('')
    
    RRS_recom_int = []
    Random_recom_int = []
    CF_item_recom_int = []
    RATED_recom_int = []
    BusOri_recom_int = []
    rankagg_recom_int = []
    rrs_usersPerproduction = {}
    random_usersPerproduction = {}
    cf_item_usersPerproduction = {}
    busOri_usersPerproduction = {}
    rankagg_usersPerproduction = {}
   
    
    for row in recom[:].itertuples():
    
        userid  = recom.loc[row.Index, 'userid'] 
        RRS_recom  = recom.loc[row.Index, 'RRS_recom']
        Random_recom  = recom.loc[row.Index, 'Random_recom']
        CF_item_recom  = recom.loc[row.Index, 'CF_item_based_recom']
        BusOri_recom  = recom.loc[row.Index, 'BusinessOriented_recom']
        rankagg_recom  = recom.loc[row.Index, 'RankAggregation_recom']
        
        if RRS_recom != '':
            RRS_recom = RRS_recom.split(",")
            RRS_recom_int = [int(j) for j in RRS_recom]
        
        if Random_recom != '':
            Random_recom = Random_recom.split(",")
            Random_recom_int = [int(j) for j in Random_recom]
            
            
        if CF_item_recom != '':
            CF_item_recom = CF_item_recom.split(",")
            CF_item_recom_int = [int(j) for j in CF_item_recom] 
           
        if BusOri_recom != '':
            BusOri_recom = BusOri_recom.split(",")
            BusOri_recom_int = [int(j) for j in BusOri_recom]     
          
        if rankagg_recom != '':
            rankagg_recom = rankagg_recom.split(",")
            rankagg_recom_int = [int(j) for j in rankagg_recom]     
            
            
        for movieid in RRS_recom_int:    
            
            productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]  
            productionid = int(productionid)
            
            if productionid in rrs_usersPerproduction.keys(): 
                existing_users = rrs_usersPerproduction.get(productionid)
                existing_users.append(userid)
                rrs_usersPerproduction[productionid] = existing_users
            else:  
                rrs_usersPerproduction[productionid] = [userid]
        
        
        for movieid in Random_recom_int:    
            
            productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]  
            productionid = int(productionid)
            
            if productionid in random_usersPerproduction.keys(): 
                existing_users = random_usersPerproduction.get(productionid)
                existing_users.append(userid)
                random_usersPerproduction[productionid] = existing_users
            else:  
                random_usersPerproduction[productionid] = [userid]
        
        
        for movieid in CF_item_recom_int: 
            
            productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]  
            productionid = int(productionid)
            
            if productionid in cf_item_usersPerproduction.keys(): 
                existing_users = cf_item_usersPerproduction.get(productionid)
                existing_users.append(userid)
                cf_item_usersPerproduction[productionid] = existing_users
            else:  
                cf_item_usersPerproduction[productionid] = [userid]
        
        for movieid in BusOri_recom_int: 
            
            productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]  
            productionid = int(productionid)
            
            if productionid in busOri_usersPerproduction.keys(): 
                existing_users = busOri_usersPerproduction.get(productionid)
                existing_users.append(userid)
                busOri_usersPerproduction[productionid] = existing_users
            else:  
                busOri_usersPerproduction[productionid] = [userid]           
                 
        for movieid in rankagg_recom_int: 
            
            productionid = movies[movies['movieId'] == movieid]['productionId_x'].iloc[0]  
            productionid = int(productionid)
            
            if productionid in rankagg_usersPerproduction.keys(): 
                existing_users = rankagg_usersPerproduction.get(productionid)
                existing_users.append(userid)
                rankagg_usersPerproduction[productionid] = existing_users
            else:  
                rankagg_usersPerproduction[productionid] = [userid]        
        
    
    return rrs_usersPerproduction , random_usersPerproduction, cf_item_usersPerproduction, busOri_usersPerproduction , rankagg_usersPerproduction


def avgUserSatisfaction(recom, ratings, knnModel_item_based, user_based, userAvgCluster, use_dcg):
    
    knnModel = knnModel_item_based
    rrs_sum_satisfaction = 0
    random_sum_satisfaction = 0
    cf_item_based_sum_satisfaction = 0
    busOri_sum_satisfaction = 0
    rankagg_sum_satisfaction = 0
    
    RRS_recom_int = []
    Random_recom_int = []
    CF_item_recom_int = []
    BusOri_recom_int = []
    rankagg_recom_int = []
    
    for row in recom[:].itertuples():
    
        userid  = recom.loc[row.Index, 'userid']
        
        RRS_recom  = recom.loc[row.Index, 'RRS_recom']
        Random_recom  = recom.loc[row.Index, 'Random_recom']
        CF_item_recom  = recom.loc[row.Index, 'CF_item_based_recom']
        BusOri_recom  = recom.loc[row.Index, 'BusinessOriented_recom']
        rankagg_recom  = recom.loc[row.Index, 'RankAggregation_recom']
        
        
        if RRS_recom != '':
            RRS_recom = RRS_recom.split(",")
            RRS_recom_int = [int(j) for j in RRS_recom]
            
        if Random_recom != '':
            Random_recom = Random_recom.split(",")
            Random_recom_int = [int(j) for j in Random_recom]    
            
        if CF_item_recom != '':
            CF_item_recom = CF_item_recom.split(",")
          
        if BusOri_recom != '':
            BusOri_recom = BusOri_recom.split(",")
            BusOri_recom_int = [int(j) for j in BusOri_recom]      
           
        if rankagg_recom != '': 
            rankagg_recom = rankagg_recom.split(",")
            rankagg_recom_int = [int(j) for j in rankagg_recom]     
            
   
        RRS_sati = getAvgRatingsRatingsPerUser(RRS_recom_int, ratings, userid, False, knnModel, userAvgCluster, use_dcg)  
        rrs_sum_satisfaction = rrs_sum_satisfaction + RRS_sati
        
        Random_sati = getAvgRatingsRatingsPerUser(Random_recom_int, ratings, userid, False, knnModel, userAvgCluster, use_dcg)  
        random_sum_satisfaction = random_sum_satisfaction + Random_sati
        
        CF_item_sati = getAvgRatingsRatingsPerUser(CF_item_recom_int, ratings, userid, False, knnModel, userAvgCluster,use_dcg)  
        cf_item_based_sum_satisfaction = cf_item_based_sum_satisfaction + CF_item_sati
       
        busOri_sati = getAvgRatingsRatingsPerUser(BusOri_recom_int, ratings, userid, False, knnModel, userAvgCluster,use_dcg)   
        busOri_sum_satisfaction = busOri_sum_satisfaction + busOri_sati
        
        rankagg_sati = getAvgRatingsRatingsPerUser(rankagg_recom_int, ratings, userid, False, knnModel, userAvgCluster,use_dcg)   
        rankagg_sum_satisfaction = rankagg_sum_satisfaction + rankagg_sati
    
    rrs_mean_satisfaction = round(rrs_sum_satisfaction/len(recom),2)
    random_mean_satisfaction = round(random_sum_satisfaction/len(recom),2)
    cf_item_mean_satisfaction = round(cf_item_based_sum_satisfaction/len(recom),2) 
    busOri_mean_satisfaction = round(busOri_sum_satisfaction/len(recom),2) 
    rankagg_mean_satisfaction = round(rankagg_sum_satisfaction/len(recom),2)  

    print( 'Average User Satisfaction for RRS = ', rrs_mean_satisfaction)
    print( 'Average User Satisfaction for Random = ', random_mean_satisfaction)
    print( 'Average User Satisfaction for item_based CF = ', cf_item_mean_satisfaction)
    print( 'Average User Satisfaction for Business-Oriented = ', busOri_mean_satisfaction)
    print( 'Average User Satisfaction for Rank Aggregation = ', rankagg_mean_satisfaction)
    
    return rrs_mean_satisfaction , random_mean_satisfaction , cf_item_mean_satisfaction, busOri_mean_satisfaction , rankagg_mean_satisfaction


def getAvgRatingsRatingsPerUser(recom_list, ratings, userid, usingPredictedRatings, knnModel, userAvgCluster, use_dcg):
    
    sum_ratings = 0
    
    if len(recom_list) != 0:
    
        for movieid in recom_list:
            
            movie_rating = ratings[(ratings['userId'] == userid) & (ratings['movieId'] == movieid)]['rating']
            
            
            if movie_rating.empty == False:
                sum_ratings = sum_ratings + movie_rating.tail(1).iloc[0] #seires
                
                """
                if use_dcg == True:
                     sum_ratings = sum_ratings + (movie_rating.tail(1).iloc[0] / np.log2( list.recom_list(movieid) + 1)) #exclude the first item!!!!
                """    
            
            else: # if user does not have rating for a movie then compute the average ratings of all users for that movie

                # average of cluster
                avgusers = userAvgCluster[userid,movieid]
                sum_ratings = sum_ratings + avgusers
                 
  
        avg_ratings =  round(sum_ratings / len(recom_list),2)
        return avg_ratings
    
    else:
        return 0        



def jaccard_similarity(list1, list2):
    
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    
    if union != 0:
        return round(float(intersection / union),2)
    else:
        return 0


def avgItemListsSimilarity(recom):
    
    rrs_rated_sum_jsim = 0
    rrs_busiOri_sum_jsim = 0
    rrs_rankagg_sum_jsim = 0
    rated_busiOri_sum_jsim = 0
    rated_rankagg_sum_jsim = 0
    busOri_rankagg_sum_jsim = 0
    
    RRS_recom_int = []
    RATED_recom_int = []
    BusOri_recom_int = []
    RERANK_recom_int = []
    rankagg_recom_int = []
    
    for row in recom[:].itertuples():
        
        RRS_recom  = recom.loc[row.Index, 'RRS_recom']
        RATED_recom  = recom.loc[row.Index, 'Rate_based_recom']
        BusOri_recom  = recom.loc[row.Index, 'BusinessOriented_recom']
        RERANK_recom  = recom.loc[row.Index, 'ReRanked_recom']
        rankagg_recom  = recom.loc[row.Index, 'RankAggregation_recom']
        
        
        if RRS_recom != '':
            RRS_recom = RRS_recom.split(",")
            RRS_recom_int = [int(j) for j in RRS_recom]
            
        if RATED_recom != '':
            RATED_recom = RATED_recom.split(",")
            RATED_recom_int = [int(j) for j in RATED_recom]      
            
        if BusOri_recom != '':
            BusOri_recom = BusOri_recom.split(",")
            BusOri_recom_int = [int(j) for j in BusOri_recom]      
            
        if RERANK_recom != '': 
            RERANK_recom = RERANK_recom.split(",")
            RERANK_recom_int = [int(j) for j in RERANK_recom] 
            
        if rankagg_recom != '': 
            rankagg_recom = rankagg_recom.split(",")
            rankagg_recom_int = [int(j) for j in rankagg_recom]     
            
            
        rrs_rated_jsim = jaccard_similarity(RRS_recom_int, RATED_recom_int)   
        rrs_rated_sum_jsim = rrs_rated_sum_jsim + rrs_rated_jsim
    
        rrs_busiOri_jsim = jaccard_similarity(RRS_recom_int, BusOri_recom_int)   
        rrs_busiOri_sum_jsim = rrs_busiOri_sum_jsim + rrs_busiOri_jsim
        
        rrs_rerank_jsim = jaccard_similarity(RRS_recom_int, RERANK_recom_int)   
        rrs_rerank_sum_jsim = rrs_rerank_sum_jsim + rrs_rerank_jsim
        
        rrs_rankagg_jsim = jaccard_similarity(RRS_recom_int, rankagg_recom_int)   
        rrs_rankagg_sum_jsim = rrs_rankagg_sum_jsim + rrs_rankagg_jsim
        
        rated_busiOri_jsim = jaccard_similarity(RATED_recom_int, BusOri_recom_int)   
        rated_busiOri_sum_jsim = rated_busiOri_sum_jsim + rated_busiOri_jsim
     
        rated_rerank_jsim = jaccard_similarity(RATED_recom_int, RERANK_recom_int)   
        rated_rerank_sum_jsim = rated_rerank_sum_jsim + rated_rerank_jsim
        
        rated_rankagg_jsim = jaccard_similarity(RATED_recom_int, rankagg_recom_int)   
        rated_rankagg_sum_jsim = rated_rankagg_sum_jsim + rated_rankagg_jsim
        
        busOri_rerank_jsim = jaccard_similarity(BusOri_recom_int, RERANK_recom_int)   
        busOri_rerank_sum_jsim = busOri_rerank_sum_jsim + busOri_rerank_jsim
        
        busOri_rankagg_jsim = jaccard_similarity(BusOri_recom_int, rankagg_recom_int)   
        busOri_rankagg_sum_jsim = busOri_rankagg_sum_jsim + busOri_rankagg_jsim
        
        rerank_rankagg_jsim = jaccard_similarity(RERANK_recom_int, rankagg_recom_int)   
        rerank_rankagg_sum_jsim = rerank_rankagg_sum_jsim + rerank_rankagg_jsim
    
    rrs_rated_mean_jsim = round(rrs_rated_sum_jsim/len(recom),2)
    rrs_busiOri_mean_jsim = round(rrs_busiOri_sum_jsim/len(recom),2) 
    rrs_rerank_mean_jsim = round(rrs_rerank_sum_jsim/len(recom),2) 
    rrs_rankagg_mean_jsim = round(rrs_rankagg_sum_jsim/len(recom),2) 
    rated_busiOri_mean_jsim = round(rated_busiOri_sum_jsim/len(recom),2)
    rated_rerank_mean_jsim = round(rated_rerank_sum_jsim/len(recom),2)
    rated_rankagg_mean_jsim = round(rated_rankagg_sum_jsim/len(recom),2)
    busOri_rerank_mean_jsim = round(busOri_rerank_sum_jsim/len(recom),2)
    busOri_rankagg_mean_jsim = round(busOri_rankagg_sum_jsim/len(recom),2)
    rerank_rankagg_mean_jsim = round(rerank_rankagg_sum_jsim/len(recom),2)
    

    print( 'Average RRS-Rate_based Jaccard Similarity = ', rrs_rated_mean_jsim)
    print( 'Average RRS-Business_Oriented Jaccard Similarity = ', rrs_busiOri_mean_jsim)
    print( 'Average RRS-Rerank Jaccard Similarity = ', rrs_rerank_mean_jsim)
    print( 'Average RRS-RankAggregation Jaccard Similarity = ', rrs_rankagg_mean_jsim)
    print( 'Average Rate_based-Business_Oriented Jaccard Similarity = ', rated_busiOri_mean_jsim)
    print( 'Average Rate_based-Rerank Jaccard Similarity = ', rated_rerank_mean_jsim)
    print( 'Average Rate_based-RankAggregation Jaccard Similarity = ', rated_rankagg_mean_jsim)
    print( 'Average Business_Oriented-Rerank Jaccard Similarity = ', busOri_rerank_mean_jsim)
    print( 'Average Business_Oriented-RankAggregation Jaccard Similarity = ', busOri_rankagg_mean_jsim)
    print( 'Average Rerank-RankAggregation Jaccard Similarity = ', rerank_rankagg_mean_jsim)
    
    
    return rrs_rated_mean_jsim, rrs_busiOri_mean_jsim, rrs_rerank_mean_jsim, rrs_rankagg_mean_jsim, rated_busiOri_mean_jsim, rated_rerank_mean_jsim, rated_rankagg_mean_jsim, busOri_rerank_mean_jsim, busOri_rankagg_mean_jsim, rerank_rankagg_mean_jsim


def dcg_at_k(r, k, method=0):
    
    r = np.asfarray(r)[:k]
    
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
            
    return 0   

def DCG(recom, k): 
    
    rrs_sum_dcg = 0
    cf_item_based_sum_dcg = 0
    rated_sum_dcg = 0
    busOri_sum_dcg = 0
    rankagg_sum_dcg = 0
    
    for row in recom[:].itertuples():
        
        RRS_scores = recom.loc[row.Index, 'RRS_scores']
        CF_scores = recom.loc[row.Index, 'CF_item_based_ratings']
        Rate_scores = recom.loc[row.Index, 'Rate_based_ratings']
        Busi_scores = recom.loc[row.Index, 'BusinessOriented_scores'] 
        
        if RRS_scores != '':
            RRS_scores = RRS_scores.split(",")
            RRS_scores_list = [float(j) for j in RRS_scores]
            
        if CF_scores != '':
            CF_scores = CF_scores.split(",")
            CF_scores_list = [float(j) for j in CF_scores]  
        
        if Rate_scores != '':
            Rate_scores = Rate_scores.split(",")
            Rate_scores_list = [float(j) for j in Rate_scores]      
        
        if Busi_scores != '':
            Busi_scores = Busi_scores.split(",")
            Busi_scores_list = [float(j) for j in Busi_scores]  
            

        rrs_dcg = dcg_at_k(RRS_scores_list, k)  
        rrs_sum_dcg = rrs_sum_dcg + rrs_dcg
        
        cf_dcg = dcg_at_k(CF_scores_list, k)  
        cf_item_based_sum_dcg = cf_item_based_sum_dcg + cf_dcg    
        
        rate_dcg = dcg_at_k(Rate_scores_list, k)  
        rated_sum_dcg = rated_sum_dcg + rate_dcg    

        busi_dcg = dcg_at_k(Busi_scores_list, k)  
        busOri_sum_dcg = busOri_sum_dcg + busi_dcg


    rrs_mean_dcg = round(rrs_sum_dcg / len(RRS_scores),2)  
    cf_mean_dcg = round(cf_item_based_sum_dcg / len(CF_scores),2)  
    rate_mean_dcg = round(rated_sum_dcg / len(Rate_scores),2)  
    busi_mean_dcg = round(busOri_sum_dcg / len(Busi_scores),2)        
        
        
    print('Average DCG for RRS = ', rrs_mean_dcg)
    print('Average DCG for CF = ', cf_mean_dcg)
    print('Average DCG for Rate-based = ', rate_mean_dcg)
    print('Average DCG for Business-based = ', busi_mean_dcg)
    #-------------------------------------
    return rrs_mean_dcg, cf_mean_dcg, rate_mean_dcg, busi_mean_dcg    

