#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:52:18 2020

@author: tahereh
"""

from recom import main, getUsersMoviesVectors, getProductionsUsersVectors
import csv
import numpy as np; np.random.seed(42)
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from SVD import knn
from dataPreprocessing import avgRatingsPerCluster



def drawSeabornBocplot():
    
    file = "../data/movielens_boxplot_final_borda_100.csv"
    df = pd.read_csv(file, sep=",")
    
    print('User Satisfaction Box Plot')
    ax = sns.boxplot(data=df.iloc[:,0:4])
    sns.set_style("ticks")
    b0 = ax.artists[0]
    b0.set_facecolor('blue')
    b1 = ax.artists[1]
    b1.set_facecolor('red')
    b2 = ax.artists[2]
    b2.set_facecolor('green')
    b3 = ax.artists[3]
    b3.set_facecolor('yellow')
      
    ax.set(ylim=(0, 6))
    plt.show()
    
    print('Business Satisfaction Box Plot')
    ax = sns.boxplot(data=df.iloc[:,4:8])
    sns.set_style("ticks")
    b0 = ax.artists[0]
    b0.set_facecolor('blue')
    b1 = ax.artists[1]
    b1.set_facecolor('red')
    b2 = ax.artists[2]
    b2.set_facecolor('green')
    b3 = ax.artists[3]
    b3.set_facecolor('yellow')
   
    ax.set(ylim=(0, 6))
    plt.show()

  
def boxplot_2d(df, ax, whis=1.5):
    
    x = df.iloc[:,0]
    y = df.iloc[:,5]
    
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0 , color='blue')
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1)
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1)
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k')
    #-------------------------------------------------
    x = df.iloc[:,1]
    y = df.iloc[:,6]
    
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0, color='red')
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1)
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1)
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k')
    #-------------------------------------------------
    x = df.iloc[:,2]
    y = df.iloc[:,7]
    
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0, color='green')
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1)
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1)
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k')
    
    
    #-------------------------------------------------
    x = df.iloc[:,3]
    y = df.iloc[:,8]
    
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0, color='yellow')
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1)
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1)
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k')
    
    #-------------------------------------------------
    
    x = df.iloc[:,4]
    y = df.iloc[:,9]
    
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0, color='purple')
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1)
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1)
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = 'k',
        zorder = 1)
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k')
    
    
def draw2DBoxplot(path):
 
    df = pd.read_csv(path, sep=",")
    
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    
    
    boxplot_2d(df,ax=ax, whis=1)
    ax.set(ylim=(0, 6))
    ax.set(xlim=(0, 6))

    ax.set_xlabel('User Satisfaction')
    ax.set_ylabel('Business Satisfaction')
    
    plt.show()


def drawXYScatterPlot():

    ## or use EXCEl
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.scatter([1,2,3,4,5], [2,3,4,5,6], s=10, c='b', marker="s", label='2RS')
    ax1.annotate('2RS_2%', (1,2), textcoords="offset points", xytext=(0,10), ha='center')
    ax1.annotate('2RS_5%', (2,3), textcoords="offset points", xytext=(0,10), ha='center')
    
    
    ax1.scatter([1.5,2.5,3.5,4.5,5.5],[2.5,3.5,4.5,5.5,6.5], s=10, c='r', marker="o", label='CF')
    plt.legend(loc='upper left');
    plt.show()        
    
    
    
if __name__ == "__main__": 
    
     path = '../data/movielens_boxplot_newMetrics_10%_1_5_4_1000_runs.csv'
     number_of_runs = 1000
     newMetrics = True

     if  newMetrics == False:
         
        iskmeans = True
        knnModel_item_based = knn(False)
        ratings = pd.read_csv('../data/ratings.csv', sep=",")
        movies = pd.read_csv('../data/movies.csv', sep=",")
        userAvgCluster = avgRatingsPerCluster(ratings,movies,iskmeans)
        
        with open(path , 'w', newline='') as file:
         
         writer = csv.writer(file)
         writer.writerow(['RRS_user_satisfaction', 'Random_user_satisfaction', 'CFI_user_satisfaction', 'Business_oriented_user_satisfaction' , 'RankAggregation_user_satisfaction',  'RRS_business_satisfaction', 'Random_business_satisfaction', 'CFI_business_satisfaction', 'Business_oriented_business_satisfaction' , 'RankAggregation_business_satisfaction'])
         for i in range(0,number_of_runs):
                
            print('RUN #',i,'***********************************')
            RRS_user_satisfaction, Random_user_satisfaction, CFI_user_satisfaction, Business_oriented_user_satisfaction , RankAggregation_user_satisfaction,  RRS_business_satisfaction, Random_business_satisfaction, CFI_business_satisfaction, Business_oriented_business_satisfaction , RankAggregation_business_satisfaction = main(knnModel_item_based,userAvgCluster, newMetrics, None, None, None, None, None, None, None, None)
            print('***********************************')
            writer.writerow([RRS_user_satisfaction, Random_user_satisfaction, CFI_user_satisfaction, Business_oriented_user_satisfaction , RankAggregation_user_satisfaction,  RRS_business_satisfaction, Random_business_satisfaction, CFI_business_satisfaction, Business_oriented_business_satisfaction , RankAggregation_business_satisfaction])    
        
     else:
        
        knnModel_item_based = knn(False)
        users = pd.read_csv('../data/users.csv', sep=",")
        ratings = pd.read_csv('../data/ratings.csv', sep=",")
        movies = pd.read_csv('../data/movies.csv', sep=",")
        productions = pd.read_csv('../data/productions.csv', sep=",")
        userids = users['userId']
        userVisits = {} 
        
        
        userPre_vectors, movie_vectors, Mcolumns = getUsersMoviesVectors(movies, users, ratings, userids, True)
        production_vectors, userAtt_vectors, userVisits, user_columns, pro_columns = getProductionsUsersVectors(movies, users, ratings, productions, userVisits, True)
        
        with open(path , 'w', newline='') as file:
         
         writer = csv.writer(file)
         writer.writerow(['RRS_user_satisfaction', 'Random_user_satisfaction', 'CFI_user_satisfaction', 'Business_oriented_user_satisfaction' , 'RankAggregation_user_satisfaction',  'RRS_business_satisfaction', 'Random_business_satisfaction', 'CFI_business_satisfaction', 'Business_oriented_business_satisfaction' , 'RankAggregation_business_satisfaction'])
         #writer.writerow(['RRS_user_satisfaction', 'CFI_user_satisfaction', 'CFU_user_satisfaction', 'Rated_user_satisfaction', 'Business_oriented_user_satisfaction' , 'RankAggregation_user_satisfaction',  'RRS_business_satisfaction', 'CFI_business_satisfaction', 'CFU_business_satisfaction', 'Rated_business_satisfaction', 'Business_oriented_business_satisfaction' , 'RankAggregation_business_satisfaction'])
         
         for i in range(0,number_of_runs):
                
            print('RUN #',i,'***********************************')
            RRS_user_satisfaction, Random_user_satisfaction, CFI_user_satisfaction, Business_oriented_user_satisfaction , RankAggregation_user_satisfaction,  RRS_business_satisfaction, Random_business_satisfaction, CFI_business_satisfaction, Business_oriented_business_satisfaction , RankAggregation_business_satisfaction = main(knnModel_item_based, None, newMetrics, userPre_vectors, movie_vectors, Mcolumns, production_vectors, userAtt_vectors, userVisits, user_columns, pro_columns) 
            #RRS_user_satisfaction, CFI_user_satisfaction, CFU_user_satisfaction, Rated_user_satisfaction, Business_oriented_user_satisfaction , RankAggregation_user_satisfaction,  RRS_business_satisfaction, CFI_business_satisfaction, CFU_business_satisfaction, Rated_business_satisfaction, Business_oriented_business_satisfaction , RankAggregation_business_satisfaction = main(knnModel_item_based,userAvgCluster)
            print('***********************************')
            writer.writerow([RRS_user_satisfaction, Random_user_satisfaction, CFI_user_satisfaction, Business_oriented_user_satisfaction , RankAggregation_user_satisfaction,  RRS_business_satisfaction, Random_business_satisfaction, CFI_business_satisfaction, Business_oriented_business_satisfaction , RankAggregation_business_satisfaction])    
        
     #drawXYScatterPlot()
     #drawSeabornBocplot()
     #draw2DBoxplot(path)
     
     
    
     
     
            
     

