#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:11:03 2020

@author: tahereh
"""

import pandas as pd
import numpy as np
from itertools import groupby


def countMoviesPerProductionCompanies():
    
    movies = pd.read_csv('../data/movies.csv', sep=",")
    results = movies[['movieId','production_company']].groupby(['production_company']).size().reset_index(name='counts')
    results.to_csv('../data/moviesPercompanies.csv', sep = ',', index= False)
    results[results['counts'] >= 5].to_csv('../data/moviesPercompanies>5.csv', sep = ',', index= False)

def userAgeDistribution(users):
    
    
    bins = [0, 20, 30, 40, 50, 60, 70]
    users['age_bins'] = pd.cut(users['age'], bins)
    age_bins = users['age_bins'].value_counts()
    print(age_bins)

 
def genderDistribution(users):
    
    gender = {'M': 1,'F': 2} 
    users.gender = [gender[item] for item in users.gender] 
    
    bins = [0,1,2]
    users['gender_bins'] = pd.cut(users['gender'], bins)
    gender_bins = users['gender_bins'].value_counts()
    print(gender_bins)
    
    
    
def agegenderDistribution(users):
    
    size = len(users)
    
    male_users = users[users['gender'] == 'M']
    female_users = users[users['gender'] == 'F']
    bins = [0, 20, 30, 40, 50, 60, 70]
    
    male_users['m_age_bins'] = pd.cut(male_users['age'], bins)
    male_age_bins = round(male_users['m_age_bins'].value_counts()*100/size,2)
    print(male_age_bins)
    female_users['f_age_bins'] = pd.cut(female_users['age'], bins)
    female_age_bins = round(female_users['f_age_bins'].value_counts()*100/size,2)
    print(female_age_bins)
    
    
    
def occupationDistribution(users):
    
    occupations = pd.read_csv('../data/occupations.csv', sep=",")
    size = len(users)
    mergedf = pd.merge(users , occupations , on=['occupationId'], how='inner')
    results = round(mergedf[['userId','occupation']].groupby(['occupation']).size()*100/size,2).reset_index(name='percent')
    print(results)
    results.to_csv('../data/occupationDist.csv', sep = ',', index= False)
 
    
    
def zipcodeDistribution(users):

    size = len(users)
    results = round(users[['userId','zipcode']].groupby(['zipcode']).size()*100/size,2).reset_index(name='percent')
    print(results)
    results.to_csv('../data/zipDist.csv', sep = ',', index= False)
 
    
    
def zipcodeDistribution2(users):

    users_per_zip = {}
    zipcodes = users['zipcode'].tolist()
    
    sorted_zip = sorted(zipcodes, key = lambda x: x[0])
    grouped_zip = [list(g) for k, g in groupby(sorted_zip, key=lambda x: x[0])]
  
    
    for i in range(0,9):
        
        users_in_group = users[users['zipcode'].isin(grouped_zip[i])]  
        user_count = len(users_in_group)
        users_per_zip[i] = user_count

    
    print(users_per_zip)
    
    
    
def zipcodeDistribution3(users):

    users_per_zip = {} 
   
    zipcodes = users['zipcode'].tolist()
    zipcodes = list(map(str, zipcodes))
    zipcodes4 = [item for item in zipcodes if len(item)==4]
    zipcodes5 = [item for item in zipcodes if len(item)==5]
    #print(len(zipcodes4), len(zipcodes5))
    
    sorted_zip4 = sorted(zipcodes4, key = lambda x: x[3])
    grouped_zip4 = [list(g) for k, g in groupby(sorted_zip4, key=lambda x: x[3])]
    #print(grouped_zip4)
    
    sorted_zip5 = sorted(zipcodes5, key = lambda x: x[4])
    grouped_zip5 = [list(g) for k, g in groupby(sorted_zip5, key=lambda x: x[4])] 
    #print(grouped_zip5)
    
    #results.to_csv('../data/zipDist.csv', sep = ',', index= False)    
    
    for i in range(0,10):
        
        users_in_group = users[users['zipcode'].isin(grouped_zip4[i])]  
        user_count4 = len(users_in_group)
        
        users_in_group = users[users['zipcode'].isin(grouped_zip5[i])]  
        user_count5 = len(users_in_group)
        
        users_per_zip[i] = user_count4 +  user_count5

    print(users_per_zip)    
    
    
def usersRatedTopComapnies(users):
    
    moviesTopcompanies = pd.read_csv('../data/moviesPercompanies>5.csv', sep=",")
    movies = pd.read_csv('../data/movies.csv', sep=",")
    rating = pd.read_csv('../data/ratings.csv', sep=",")
    movies_top_companies = movies[movies['production_company'].isin(moviesTopcompanies['production_company'])]
    users_top_comapnies = rating[rating['movieId'].isin(movies_top_companies['movieId'])]['userId']
    users_top_comapnies = users_top_comapnies.unique()
    print(len(users_top_comapnies))
    

def ratingsPerUser():
    
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    results = ratings[['userId']].groupby(['userId']).size().reset_index(name='rating_count')
    results.to_csv('../data/ratingsPerUser.csv', sep = ',', index= False)


    from matplotlib import pyplot as plt
    plt.hist(ratings['rating'], bins=5, ec='black')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Ratings in MovieLens 100K')
    plt.show()
    
    

def ratingsPerMovie():
    
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    results = ratings[['movieId']].groupby(['movieId']).size().reset_index(name='rating_count')
    results.to_csv('../data/ratingsPerMovie.csv', sep = ',', index= False)
    
   
    
    

if __name__ == "__main__":     
    
    ratingsPerUser()
    ratingsPerMovie()

    
    
    
    
    
    