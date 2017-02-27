"""
Decision Tree Modeling
1. Get features
    1) Autocorrelation analysis for feature selection
    2) Weekly and daily circadian rhythmicity of physical activity
2. Train DT
3. Make prediction
"""
# TODO: expand DT to random forest

# 1. get features

from __future__ import print_function
import configparser
import datetime as dt
import glob
import itertools
import operator
import numpy as np
import pandas as pd
import pylab as plt
import statsmodels.tsa.stattools as smtsa
from bokeh.plotting import figure, output_file  # output to static HTML file
from scipy.optimize import leastsq
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot
import copy


class basicModel():
    def __init__(self, data, feature, target, model_method):
        self.data = data
        self.feature = feature
        self.target = target
        self.model_method = model_method

    def train_model(self, feature_selected):
        y = self.data[self.target]
        x = self.data[feature_selected]
        model = self.model_method
        model = model.fit(x, y)
        train_accuracy = model.score(x, y)
        print("accuracy based on training data is %f" % train_accuracy)
        return model, feature_selected, train_accuracy


class optimizedModel(basicModel):
    # def __init__(self):
    #     basicModel.__init__(self, data, feature, target, model_method)
    #     self.optimized = optimized

    @staticmethod
    def get_combination(lst):
        combination_list = []
        for i in range(1, len(lst) + 1):
            features_tuple = itertools.combinations(lst, i)
            for feature_combination in features_tuple:
                combination_list.append(list(feature_combination))
        return combination_list

    @staticmethod
    def get_max(lst):
        max_index, max_value = max(enumerate(lst), key=operator.itemgetter(1))
        return max_index, max_value

    def optimize_model(self):
        features_c_lst = self.get_combination(self.feature)
        scores = []
        models = []
        for features in features_c_lst:
            current_model, current_feature, current_score = self.train_model(features)
            scores.append(current_score)
            models.append(current_model)
        max_score_index, max_score = self.get_max(scores)
        best_model = models[max_score_index]
        best_features = features_c_lst[max_score_index]
        print('best features are: %s' % str(best_features).strip('[]'))
        print("accuracy based on training data is %f" % max_score)
        return best_model, best_features


def get_prediction(data, model, target, feature):
    print(feature)
    x_predict = data[feature]
    y_predicted = model.predict(x_predict)
    # data.loc[:, 'predicted'] = y_predicted
    accuracy = model.score(data[feature], data[target])
    print("prediction accuracy based on test data is %f " % accuracy)
    return y_predicted, accuracy


def evaluate_model(accuracy):
    if accuracy >= .9:
        print('model adopted')
        return True
    else:
        print('current model does not meet accuracy threshold, therefore rejected.')
        return False


def get_dt_model(data, feature, target, optimized=True):
    model_method = DecisionTreeClassifier(min_samples_split=30)
    if optimized:
        DtModel = optimizedModel(data, feature, target, model_method)
        train_dt_model, train_features = DtModel.optimize_model()
    else:
        DtModel = basicModel(data, feature, target, model_method)
        train_dt_model, train_features = DtModel.train_model(feature)
    return train_dt_model, train_features



# def get_tree_model(data, target, manual_select=[], optimize=True):
#     features_lst_all = data.columns.values[4:]
#     features_lst = []
#     for i in features_lst_all:
#         if i != 'inactive':
#             features_lst.append(i)
#     if optimize:
#         features_c_lst = get_feature_combinations(features_lst)
#         scores = []
#         trees = []
#         for features in features_c_lst:
#             y = data[target]
#             x = data[features]
#             # print(data.head())
#             tree = DecisionTreeClassifier(min_samples_split=30)
#             tree = tree.fit(x, y)
#             accuracy = tree.score(x, y)
#             scores.append(accuracy)
#             trees.append(tree)
#         max_score_index, max_score = get_max(scores)
#         best_tree = trees[max_score_index]
#         best_features = features_c_lst[max_score_index]
#         print('best features are: %s' % str(best_features).strip('[]'))
#         print("accuracy based on training data is %f" % max_score)
#         return best_tree, best_features
#     else:
#         features_lst = features_lst[manual_select]
#         y = data[target]
#         x = data[features_lst]
#         # print(data.head())
#         tree = DecisionTreeClassifier(min_samples_split=30)
#         tree = tree.fit(x, y)
#         accuracy = tree.score(x, y)
#         print("accuracy based on training data is %f" % accuracy)
#         return tree, features_lst

