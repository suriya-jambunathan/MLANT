#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:06:35 2021

@author: suriyaprakashjambunathan
"""

import warnings
warnings.simplefilter(action='ignore')

#Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

#Regressors
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from  sklearn.isotonic import IsotonicRegression
from sklearn.linear_model.bayes import ARDRegression
from sklearn.linear_model.huber import HuberRegressor
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor 
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.linear_model.theil_sen import TheilSenRegressor
from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.neighbors.regression import RadiusNeighborsRegressor
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.tree.tree import ExtraTreeRegressor
from sklearn.svm.classes import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.svm import NuSVR
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

# Importing the Libraries
import pandas as pd
import numpy as np



Name_c = ['BaggingClassifier',
         'BernoulliNB',
         'CalibratedClassifierCV',
         'ComplementNB',
         'DecisionTreeClassifier',
         'DummyClassifier',
         'ExtraTreeClassifier',
         'ExtraTreesClassifier',
         'GaussianNB',
         'GaussianProcessClassifier',
         'GradientBoostingClassifier',
         'HistGradientBoostingClassifier',
         'KNeighborsClassifier',
         'LabelPropagation',
         'LabelSpreading',
         'LinearDiscriminantAnalysis',
         'LinearSVC',
         'LogisticRegression',
         'LogisticRegressionCV',
         'MLPClassifier',
         'MultinomialNB',
         'NearestCentroid',
         'PassiveAggressiveClassifier',
         'Perceptron',
         'QuadraticDiscriminantAnalysis',
         'RadiusNeighborsClassifier',
         'RandomForestClassifier',
         'RidgeClassifier',
         'RidgeClassifierCV',
         'SGDClassifier',
         'SVC']


Name_r = [ "RandomForestRegressor",
        "ExtraTreesRegressor",
        "BaggingRegressor",
        "GradientBoostingRegressor",
        "AdaBoostRegressor",
        "GaussianProcessRegressor",
        "ARDRegression",
        "HuberRegressor",
        "LinearRegression",
        "PassiveAggressiveRegressor",
        "SGDRegressor",
        "TheilSenRegressor",
        "KNeighborsRegressor",
        "RadiusNeighborsRegressor",
        "MLPRegressor",
        "DecisionTreeRegressor",
        "ExtraTreeRegressor",
        "SVR",
        "BayesianRidge",
        "CCA",
        "ElasticNet",
        "ElasticNetCV",
        "KernelRidge",
        "Lars",
        "LarsCV",
        "Lasso",
        "LassoCV",
        "LassoLars",
        "LassoLarsIC",
        "LassoLarsCV",
        "NuSVR",
        "OrthogonalMatchingPursuit",
        "OrthogonalMatchingPursuitCV",
        "PLSCanonical",
        "Ridge",
        "RidgeCV",
        "LinearSVR"]

class models(object):
    
    @classmethod
    def classifiers(self, X_train,y_train):
        clfs = []
        Name = Name_c
        for i in range(len(Name)):
            classifier = globals()[Name[i]]
            #print(classifier)
            Classifier = classifier()
            Classifier.fit(X_train, y_train)
            clfs.append(Classifier)
        return(clfs)      
        
    @classmethod
    def regressors(self, X_train,y_train):
        regs = []
        Name = Name_r
        for i in range(len(Name)):
            regressor = globals()[Name[i]]
            #print(regressor)
            Regressor = regressor()
            Regressor.fit(X_train, y_train)
            regs.append(Regressor)
        return(regs)
    