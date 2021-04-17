#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:08:57 2021

@author: suriyaprakashjambunathan
"""

import numpy as np
import random
from sklearn.metrics import mean_squared_error
# GENETIC ALGORITHM

class GeneticAlgorithm(object):
    
    def __init__(self,
                 X_train,X_test, 
                 y_train, y_test,
                 size,n_feat,n_parents,
                 mutation_rate,n_gen,
                 model):
        
        # Dataframes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Algorithm Configurations
        self.size = size
        self.n_feat = n_feat
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.n_gen = n_gen
        
        # Classification/Regression Model 
        self.model = model
        
        self.best_features = None
    def fit(self):
        # Dataframes
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        
        # Algorithm Configurations
        size = self.size
        n_feat = self.n_feat
        n_parents = self.n_parents
        mutation_rate = self.mutation_rate
        n_gen = self.n_gen
        
        # Classification/Regression Model 
        model = self.model
        
        best_chromo= []
        best_score= []
        population_nextgen = ga_processing.pop_initialize(size,n_feat)
        gen_count = 0
        for i in range(n_gen):
            gen_count = gen_count + 1
            scores, pop_after_fit = ga_processing.fitness_score(population_nextgen, 
                                                                X_train, y_train,
                                                                X_test, y_test,
                                                                model)
            print(" Generation "  + str(gen_count))
            print(scores[:2])
            print(' ')
            pop_after_sel = ga_processing.selection(pop_after_fit,n_parents)
            pop_after_cross = ga_processing.crossover(pop_after_sel)
            population_nextgen = ga_processing.mutation(pop_after_cross,mutation_rate)
            best_chromo.append(pop_after_fit[0])
            best_score.append(scores[0])
            
        self.best_features = best_chromo[best_score.index(min(best_score))]
        
        return best_chromo,best_score
        
        
#defining various steps required for the genetic algorithm
class ga_processing(object):
    
    @classmethod
    def pop_initialize(self, size,n_feat):
        population = []
        for i in range(size):
            chromosome = np.ones(n_feat,dtype=np.bool)
            chromosome[:int((random.uniform(0.2,0.7))*n_feat)]=False
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population
    
    @classmethod
    def fitness_score(self, population, X_train, y_train, X_test, y_test, model):
        scores = []
        for chromosome in population:
            try:
                model.fit((X_train.iloc[:,chromosome]),(y_train))
                predictions = model.predict((X_test.iloc[:,chromosome]))
                scores.append(mean_squared_error(y_test),list(predictions))
            except:
                scores.append(1000000000)
        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)
        return list(scores[inds]), list(population[inds,:])
    
    @classmethod
    def selection(self, pop_after_fit,n_parents):
        population_nextgen = []
        for i in range(n_parents):
            population_nextgen.append(pop_after_fit[i])
        return population_nextgen
    
    @classmethod
    def crossover(self, pop_after_sel):
        population_nextgen=pop_after_sel
        for i in range(len(pop_after_sel)):
            child=pop_after_sel[i]
            child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
            population_nextgen.append(child)
        return population_nextgen
    
    @classmethod
    def mutation(self, pop_after_cross,mutation_rate):
        population_nextgen = []
        for i in range(0,len(pop_after_cross)):
            chromosome = pop_after_cross[i]
            for j in range(len(chromosome)):
                if random.random() < mutation_rate:
                    chromosome[j]= not chromosome[j]
            population_nextgen.append(chromosome)
        #print(population_nextgen)
        return population_nextgen


