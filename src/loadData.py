#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:52:43 2021

@author: tahereh
"""

import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from pandas import DataFrame


def convertTocsv():
    
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
    users.to_csv('../data/users.csv', sep="," , index = False)
    
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
    ratings.to_csv('../data/ratings.csv', sep="," , index = False)

    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1')
    movies.to_csv('../data/movies.csv', sep="," , index = False)
    

def joinMovieLensIMDB():
    
    movies = pd.read_csv('../data/movies.csv', sep=",")
    movies = movies.drop_duplicates(subset ="title", keep = 'first', inplace = False) 
    movies.title = movies.title.str[:-7]
    imdb_movies = pd.read_csv('../data/IMDb movies.csv', sep=",")
    imdb_movies = imdb_movies.drop_duplicates(subset ="title", keep = 'first', inplace = False)
    mergedf = pd.merge(movies , imdb_movies , on=['title'], how='inner')
    mergedf['production_company'].replace('', np.nan, inplace=True)
    mergedf.dropna(subset=['production_company'], inplace=True)
    mergedf.to_csv('../data/movies_merged.csv', sep="," , index = False)

def filterRatings():
    
    ratings = pd.read_csv('../data/ratings.csv', sep=",")
    movies_merged = pd.read_csv('../data/movies_merged.csv', sep=",")
    
    new_ratings = ratings[ratings['movieId'].isin(movies_merged['movieId'])]
    print(len(new_ratings))
    new_ratings.to_csv('../data/ratings_filtered.csv', sep="," , index = False)
    
    
def getProductionCompanies():
    
    movies_merged = pd.read_csv('../data/movies_merged.csv', sep=",")
    production_companies = movies_merged['production_company']
    production_companies = production_companies.unique()
    productions = pd.DataFrame(columns = ['production_company']) 
    productions['production_company'] = production_companies
    productions.to_csv('../data/productions.csv', sep="," )
    
def joinProductionsToMovies():
    
    productions = pd.read_csv('../data/productions.csv', sep=",")
    movies_merged = pd.read_csv('../data/movies_merged.csv', sep=",")
    
    mergedf = pd.merge(movies_merged , productions , on=['production_company'], how='inner')
    mergedf.to_csv('../data/movies_merged.csv', sep="," , index = False)

    
def joinUserOccupations():
    
    users = pd.read_csv('../data/users.csv', sep=",")
    occupations = pd.read_csv('../data/occupations.csv', sep=",")
    mergedf = pd.merge(users , occupations , on=['occupation'], how='inner')
    mergedf.to_csv('../data/users_merged.csv', sep="," , index = False)


def BernoulliGenerator():

    p = 0.05 #2,5,10,25,50,100% : x%of productions enforce preferences
    generate = bernoulli.rvs(p, size=487)
    listg = list(generate)
    df = DataFrame (listg,columns=['column'])
    df.to_csv('../data/bernoulli.csv', sep="," , index = False)

if __name__ == "__main__":     
    
    print()
    #convertTocsv()
    #joinMovieLensIMDB()
    #filterRatings()
    #getProductionCompanies()
    #joinProductionsToMovies()
    #joinUserOccupations()
    #BernoulliGenerator()
    
    
    
    
    
    
    
    
    
    