#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:12:23 2019
@author: tahereh
"""


from rankagg import FullListRankAggregator
#from computeDiversityWeights import finalWeights
import pandas as pd


def borda_aggregation(scorelist):
    FLRA = FullListRankAggregator()
    aggRanks = FLRA.aggregate_ranks(scorelist,areScores=True,method='borda')
    #print("Borda:", aggRanks.keys(), "\n")
    return list(aggRanks.keys())
    


def footrule_aggregation(scorelist):
    
    FLRA = FullListRankAggregator()
    aggRanks = FLRA.aggregate_ranks(scorelist,areScores=True,method='spearman')
    """
    aggregate_ranks calls footrule_aggregation itself
    aggRanks = FLRA.footrule_aggregation(scorelist)
    import operator
    import collections
    sorted_x = sorted(aggRanks.items(), key=operator.itemgetter(1),reverse=True)
    sorted_dict = collections.OrderedDict(sorted_x)
    #print("Footrule:", aggRanks.keys(), "\n")
    #print(sorted_dict.keys(), sorted_dict.values() )
    #print(aggRanks.keys(), aggRanks.values())
    """
    return list(aggRanks.keys())
   


def local_kemenization(scorelist):
    FLRA = FullListRankAggregator()
    aggRanks = FLRA.aggregate_ranks(scorelist,areScores=True,method='spearman')
    ranklist = [FLRA.convert_to_ranks(s) for s in scorelist]
    lkRanks = FLRA.locally_kemenize(aggRanks,ranklist)
    #print("local kemeny:", lkRanks.keys(), "\n")
    return list(lkRanks.keys())
    
    

def highest_rank(scorelist):
    FLRA = FullListRankAggregator()
    aggRanks = FLRA.aggregate_ranks(scorelist,areScores=True,method='highest')
    return list(aggRanks.keys())

"""
if __name__ == '__main__':
    
    #example
    #scorelist = [{'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.5, 'E': 0.25, 'F': 0.5 , 'G': 0.5, 'H': 0.5, 'I': 1.0, 'J': 1.0 } , 
             #{'A': 0.33, 'B': 0.33 , 'C': 0.66 , 'D': 2.0 , 'E': 0.33 , 'F': 0.66 , 'G': 0.33 , 'H': 0.33 , 'I': 0.66 , 'J': 0.33 }]

    
    age_weights , gender_weights = finalWeights()
    age_list = age_weights[['ID', 'age_weight']]
    gender_list = gender_weights[['ID', 'gender_weight']]
    #--------------
    age_dict = dict(zip(list(age_list['ID']), list(age_list['age_weight'])))
    gender_dict = dict(zip(list(gender_list['ID']), list(gender_list['gender_weight'])))
    scorelist = []
    scorelist.append(age_dict)
    scorelist.append(gender_dict)
    print("scorelist:", scorelist, "\n")
    #--------------
    test_borda_aggregation(scorelist)
    test_footrule_aggregation(scorelist)
    test_local_kemenization(scorelist)
    psl_rankagg(age_list, gender_list)
 """  
