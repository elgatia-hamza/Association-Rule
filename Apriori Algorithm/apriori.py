# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:13:30 2020

@author: Hamza
"""


# Association-Rule Learning 
# Apriori-Algorithm

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Data Preprocessing
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])
    
# Training The Apiori algorithm on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_left = 3, min_length = 2 )

# Visualisation of the results
results = {}
list_rules = list(rules)
for rule in list_rules:
    results['->'.join(list(rule[0]))] = {'ARIORI-SUPPORT':round(rule[1], 5) ,
                             'ARIORI-CONFIDENCE:':round(rule[2][0][2]),
                            }
    
