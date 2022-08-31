#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:22:50 2020

@author: tahereh
"""

import pandas as pd
import operator
import numpy as np


def getUserAcceptedMovie(random_user, offered_movieids, isDeterministic, movies):
    
    random_movie = np.random.choice(offered_movieids, 1, replace=False)
    return random_movie[0]

