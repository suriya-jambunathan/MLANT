#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:35:01 2021

@author: suriyaprakashjambunathan
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from src.models import *
from sklearn.metrics import accuracy_score,mean_squared_error
from tqdm import tqdm
import random
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
import statistics 

class class_reg(object):
    """ A function combining classifiers and regressors"""
    def __init__(self, data = None,X_cols = None, y_col = None, test_size = 0.3,validation_size = 0.2, epochs = 5, metrics = 'wmape'):
        self.data = data
        self.X_cols = X_cols
        self.y_col = y_col
        self.test_size = test_size
        self.validation_size = validation_size
        self.epochs = epochs
        self.metrics = metrics
        self.test_X = None
        self.test_y = None
        self.classifier = None
        self.regressor = None
        self.mets = None
        
    def fitted(self):
        data = self.data
        X_cols = self.X_cols
        y_col = self.y_col
        test_size = self.test_size
        validation_size = self.validation_size
        epochs = self.epochs
        metrics = self.metrics
        mape_vals = []
        epoch_num = 0
        
        X = data[X_cols]
        y = pd.DataFrame(data[y_col])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0) 
        
        y_test = list(processing.avgfit(list(y_test[y_col])))
        dataset = []
        for i in range(len(X_train)):
            dataset.append(list(X_train.iloc[i]))
        dataset = pd.DataFrame(dataset)
        cols = []
        for i in X_cols :
            cols.append(i)
        cols.append(y_col)
        
        dataset[y_col] = y_train
        dataset.columns = cols

        self.test_X = X_test
        self.test_y = y_test
        self.train_X = X_train
        self.train_y = y_train
        for random_state in np.random.randint(0, 10000, epochs):   
            epoch_num = epoch_num + 1
            X,y,n_classes = processing.split(dataset,X_cols,y_col)
            X_train, X_test, y_train, y_test = processing.train_test(X, 
                                                          y, 
                                                          validation_size, 
                                                          random_state) 
            
            X_train_list, y_train_list = processing.dataset_split_class(X_train ,
                                                                        y_train[1],
                                                                        y,
                                                                        len(y),
                                                                        n_classes,
                                                                        'train')
            
            epoch = str(epoch_num) + '/' + str(epochs)
            print(' ')
            print("Epoch " + epoch + ' :')
            acc_conf, clf, reg = training.train(y, y_col,
                                      X_train, y_train, 
                                      X_train_list, y_train_list, 
                                      X_test, y_test, 
                                      n_classes, random_state, metrics)
            for acc_ in acc_conf:
                mape_vals.append(acc_)
        
        
        acc_vals, c, r = analysis.analyse(mape_vals,epochs)
        
        self.acc_vals = acc_vals
        
        classifier = clf[c]
        regressor = []
        for i in range(n_classes):
            regressor.append(reg[i][r])
        X_train = self.train_X
        y_train = self.train_y 
        
        train = X_train
        train[y_col] = y_train
        
        X_train,y_train,n_classes = processing.split(train,X_cols,y_col)
        #y_train = pd.DataFrame(processing.avgfit(list(y_train[y_col])))
        
        classifier.fit(X_train, pd.DataFrame(y_train[1]))
        
        X_train = processing.rem_col_name(X_train)
        y_train = processing.rem_col_name(y_train)
        
        X_train.columns = X_cols
        #y_train.columns = [y_col]
        X_train_list, y_train_list = processing.dataset_split_class(X_train ,
                                                                    y_train[1],
                                                                    y,
                                                                    len(y),
                                                                    n_classes,
                                                                    'train')
        
        for i in range(n_classes):
            (regressor[i]).fit(X_train_list[i],y_train_list[i][0])
        
        self.classifier = classifier
        self.regressor = regressor
        self.n_classes = n_classes
        #return(mape_vals, acc_vals)
        #return(classifier, regressor)
        
    def fit(self, X, y, validation_size = 0.3, epochs = 1):
        X_cols = X.columns
        y_col = y.columns
        X = processing.rem_col_name(X)
        y = processing.rem_col_name(y)
        X.columns = X_cols
        y.columns = y_col
                
        dataset = X
        dataset[y_col] = y
        epoch_num = 0
        mape_vals = []
        for random_state in np.random.randint(0, 10000, epochs):   
            epoch_num = epoch_num + 1
            X,y,n_classes = processing.split(dataset,X_cols,y_col)
            X_train, X_test, y_train, y_test = processing.train_test(X, 
                                                          y, 
                                                          validation_size, 
                                                          random_state) 
            
            X_train_list, y_train_list = processing.dataset_split_class(X_train ,
                                                                        y_train[1],
                                                                        pd.DataFrame(y),
                                                                        len(y),
                                                                        n_classes,
                                                                        'train')
            
            epoch = str(epoch_num) + '/' + str(epochs)
            print(' ')
            print("Epoch " + epoch + ' :')
            metrics = 'wmape'
            acc_conf, clf, reg = training.train(y, y_col,
                                      X_train, y_train, 
                                      X_train_list, y_train_list, 
                                      X_test, y_test, 
                                      n_classes, random_state, metrics)
            for acc_ in acc_conf:
                mape_vals.append(acc_)
        
        
        acc_vals, c, r = analysis.analyse(mape_vals,epochs)
        
        self.acc_vals = acc_vals
        
        classifier = clf[c]
        regressor = []
        for i in range(n_classes):
            regressor.append(reg[i][r])
            
        X_train,y_train,n_classes = processing.split(dataset,X_cols,y_col)

        classifier.fit(X_train, pd.DataFrame(y_train[1]))
        
        X_train.columns = X_cols
        #y_train.columns = [y_col]
        X_train_list, y_train_list = processing.dataset_split_class(X_train ,
                                                                    y_train[1],
                                                                    y,
                                                                    len(y),
                                                                    n_classes,
                                                                    'train')
        
        for i in range(n_classes):
            (regressor[i]).fit(X_train_list[i],y_train_list[i][0])
        
        self.classifier = classifier
        self.regressor = regressor
        self.n_classes = n_classes
        #return(mape_vals, acc_vals)
        #return(classifier, regressor)
        
    def predict(self, X):
        clf = self.classifier
        reg = self.regressor
        
        if isinstance(X, pd.DataFrame):  
            pred = []
            for i in range(len(X)):
                arr = list(X.iloc[i])
                pred.append(class_reg.pred(clf,reg,arr))
        else:
            X = ((np.array(X).reshape(1,-1)))
            
            clf_pred = (clf.predict(X))[0]
            
            class_ = ([int(s) for s in clf_pred.split() if s.isdigit()])[0]
            
            pred = (reg[class_ - 1].predict(X))[0]
        
        return(pred)
        
    @classmethod
    def pred(self,clf,reg,X):
        X = ((np.array(X).reshape(1,-1)))
            
        clf_pred = (clf.predict(X))[0]
        
        class_ = ([int(s) for s in clf_pred.split() if s.isdigit()])[0]
        
        pred = (reg[class_ - 1].predict(X))[0]
        
        return(pred)
        
    def performance(self):
        clf = self.classifier
        reg = self.regressor
        data = self.data
        X_cols = self.X_cols
        y_col = self.y_col
        test_size = self.test_size
        
        X,y,n_classes = processing.split(data,X_cols,y_col)
        
        mape_list = []
        mse_list = []
        
        for random_state in np.random.randint(0, 10000, 20):
            X_train, X_test, y_train, y_test = processing.train_test(X, 
                                                          y, 
                                                          test_size, 
                                                          random_state) 
            
            X_train_list, y_train_list = processing.dataset_split_class(X_train ,
                                                                        y_train[1],
                                                                        y,
                                                                        len(y),
                                                                        n_classes,
                                                                        'train')
            
            classi  = clf
            classi.fit(X_train, y_train[1])
            regr = []
            for i in range(n_classes):
                regre_ = reg[i]
                regre_.fit(X_train_list[i],y_train_list[i][0])
                regr.append(regre_)
            
            pred = []
            for i in range(len(X_test)):
                arr = list(X_test.iloc[i])
                pred.append(class_reg.pred(classi, regr, arr))
            
            mape = metric.wmape(list(y_test[0]), list(pred))
            
            mse = mean_squared_error(list(y_test[0]), pred, squared = False)
            mse = (np.sqrt(mse) - min(y_test[0]))/((max(y_test[0])) - min(y_test[0]))
            mse = mse**2
            
            mape_list.append(mape)
            mse_list.append(mse)
        
        mape = sum(mape_list)/len(mape_list)
        mse = sum(mse_list)/len(mse_list)
        
        mets = {'WMAPE' : mape, 'MSE' : mse}
        
        self.mets = mets
        return(mets)
            
            
        
        
class processing(object):
            
    @classmethod
    def avgfit(self,l):
        self.l = l
        
        na = pd.isna(l)
        arr = []
        for i in range(len(l)):
            if na[i] == False:
                arr.append(l[i])
        
        #avg = sum(arr)/len(arr)
        avg = statistics.median(arr)
        fit_arr = []
        
        for i in range(len(l)):
            if na[i] == False:
                fit_arr.append(l[i])
            elif na[i] == True:
                fit_arr.append(avg)
        
        self.fit_arr = fit_arr
        
        return(fit_arr)
        
    @classmethod
    def class_split(self,l,l_):
        self.l = l
        self.l = l_
        
        length = len(l_)

        if length <= 1000:
            n_classes = 5
        elif length <= 10000:
            n_classes = 10
        else:
            n_classes = 100
                    
        class_size = int(length/n_classes)
        
        indices = []
        for i in l:
            indices.append(l_.index(i))
            
        indices = list(np.argsort(l))

        c_list = []
        for j in range(1,n_classes+1):
            for i in range(class_size):
                c_list.append('Class ' + str(j))
                
        l_diff = length - len(c_list)
        for i in range(l_diff):
            c_list.append(c_list[-1])
            
        class_list = []
        for i in indices:
            class_list.append(c_list[i])
        
        return(class_list,n_classes)
        
    @classmethod
    def class_weight(self,arr):
        count = [(list(arr)).count(x) for x in list(set(list(arr)))]
        class_weights = dict(zip(list(set(list(arr))),count))
        
        return(class_weights)
    
    @classmethod
    def dataset_split_class(self,X,y,Y,size,n_classes,mode):
        l = [[] for _ in [None] * n_classes]
        
        if mode == 'train' :
            for i in range(size):
                    try:
                        yy = y[i]
                        ind = ([int(s) for s in yy.split() if s.isdigit()])[0]
                        l[ind - 1].append(i)
                    except:
                        continue
        elif mode == 'test':
            for i in range(size):
                    try:
                        yy = y[0][i]
                        ind = ([int(s) for s in yy.split() if s.isdigit()])[0]
                        l[ind - 1].append(y[1][i])
                    except:
                        continue
                
        X_ = []
        for i in range(n_classes):
            X_.append(X.loc[l[i]])
            
        
        y_ = []
        for i in range(n_classes):
            y_.append(Y.loc[l[i]])
        
        return(X_, y_) 
    
    @classmethod
    def rem_col_name(self, df):
        arr = []
        for i in range(len(df)):
            arr.append(list(df.iloc[i]))
        return(pd.DataFrame(arr))
        
    @classmethod
    def tolist(self, df):
        df_ = []
        for i in range(len(df)):
            df_.append((df.iloc[i])[0])
        return(df_)
    @classmethod
    def split(self,dataset,X_cols,y_col):
        try:
            y_d = self.tolist(dataset[y_col])
        except:
            y_d = list(dataset[y_col])
        X = dataset[X_cols]
        y,n_classes = self.transform(y_d)
        
        return(X,y,n_classes)
        
    @classmethod
    def train_test(self,X,y,test_size = 0.3, random_state = 0):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state) 
        
        return(X_train,X_test,y_train,y_test)
        
    @classmethod
    def transform(self,data):
        self.data = data
        
        l = data
        l = self.avgfit(l)
        l_ = sorted(l)
        
        c_list,n_classes = self.class_split(l,l_)
        
        y = pd.DataFrame()
        y[0] = l
        y[1] = c_list
        
        return(y,n_classes)   
        

class metric(object):
    
    @classmethod
    def wmape(self,y_true, y_pred):
        y_true, y_pred = list(y_true),list(y_pred)
        l = len(y_true)
        num = 0
        den = 0
        for i in range(l):
            num = num + (abs(y_pred[i] - y_true[i]))
            den = den + y_true[i]
        return abs(num/den) * 100
    
    @classmethod
    def rmse(self,y_true,y_pred):
        y_true, y_pred = list(y_true),list(y_pred)
        mse = mean_squared_error(y_true, y_pred, squared = False)
        return(mse)
        
    @classmethod
    def me_ae(self, y_true, y_pred):
        y_true, y_pred = list(y_true),list(y_pred)
        med_ae = median_absolute_error(y_true, y_pred)
        return(med_ae)
    
    @classmethod
    def mae(self, y_true, y_pred):
        y_true, y_pred = list(y_true),list(y_pred)
        mae_ = mean_absolute_error(y_true, y_pred)
        return(mae_)
        
class training(object):
    
    @classmethod
    def train(self, y, y_col, X_train, y_train, X_train_list, y_train_list, X_test, y_test, n_classes, random_state, metrics):
        Classifiers = models.classifiers(X_train,pd.DataFrame(y_train[1]))
        Regressors = []
        for i in range(n_classes):
            Regressors.append(models.regressors(X_train_list[i],pd.DataFrame(y_train_list[i][0])))
        Regressors_ = list(map(list, itertools.zip_longest(*Regressors, fillvalue=None)))
    
        acc_conf = []
        for clf in tqdm(Classifiers, leave = True):
            try:
                classifier = clf
                # Predicting the Test set results
                y_pred = pd.DataFrame(list(classifier.predict(X_test)))
                acc = accuracy_score(list(y_test[1]), list(y_pred[0]))
                
                xtestix = X_test.index.values.tolist()
                y_pred[1] = xtestix
                
                X_test_list, y_test_list = processing.dataset_split_class(X_test ,
                                                                          y_pred,
                                                                          y,
                                                                          len(X_test),
                                                                          n_classes,
                                                                          'test')
                #test_y = []
                #for i in range(n_classes):
                #   for j in list(y_test_list[i]):
                #       test_y.append(j)
                        
                
                for regr in Regressors_:
                    #pred_y = []
                    met = 0
                    for i in range(n_classes):
                        reg = regr[i]
                        y_pred_r = reg.predict(X_test_list[i])
                        #for j in list(y_pred_r):
                        #   pred_y.append(j)
                        class_reg = eval("metric." + str(metrics) + "(y_test_list[i][0],y_pred_r)")
                        met = met + class_reg*len(y_test_list[i][0])
                    met = met/len(y_test)
                    #met = eval("metric." + str(metrics) + "(test_y,pred_y)")
                    try:
                        met = met[0]
                    except:
                        met = met
                    c_loop = Classifiers.index(clf)
                    r_loop = list(map(list, itertools.zip_longest(*Regressors_, fillvalue=None)))[0].index(regr[0])
                    acc_conf.append([random_state,y_col,c_loop,r_loop,met])
                    #print(str(c_loop) + ' , ' + str(r_loop))
            except:
                continue
        
        return(acc_conf, Classifiers, Regressors)
   
    
class analysis(object):
    
    @classmethod
    def analyse(self, df,epochs):
        df = pd.DataFrame(df)
        i_arr = list(set(list(df[2])))
        j_arr = list(set(list(df[3])))
        
        acc_vals = [[[0] for _ in [None] * len(j_arr)] for _ in [None] * len(i_arr)]
        epochs = 3
        for i in i_arr:
            for j in j_arr:
                summ = 0
                for x in range(len(df)):
                    if (df[2][x] == i) and (df[3][x] == j):
                        summ = summ + df[4][x]
                        #acc_vals[i_arr.index(i)][j_arr.index(j)]= acc_vals[i_arr.index(i)][j_arr.index(j)] + meta[4][x]
                summ = summ/epochs
                acc_vals[i_arr.index(i)][j_arr.index(j)] = summ
        
        acc_vals = pd.DataFrame(acc_vals)
        
        clf_ind = acc_vals.min(axis=1).idxmin()
        reg_ind = acc_vals.min().idxmin()
        
        return(acc_vals, clf_ind, reg_ind)
        