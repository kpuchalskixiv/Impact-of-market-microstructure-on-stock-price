import pandas as pd
from torch import nn
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scs
from sklearn.preprocessing import normalize as norm
from sklearn import linear_model
import datetime
import time
from datetime import datetime , timedelta

import zipfile
import math

from csv import reader
from matplotlib.pyplot import figure
import seaborn as sns;
import re
import torch

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from sklearn.base import BaseEstimator, TransformerMixin

from market_state import *

data = np.zeros(shape = (100, 5100,2,2,10))
# sklearn Normalizer does not allow normalizing by axis!=0, and we want to normalize features->columns
class MyNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, norm="l2", axis=1, *, copy=True):
        self.norm = norm
        self.copy = copy
        self.axis=axis

    def fit(self, X, y=None):
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        return preprocessing.normalize(X, norm=self.norm, axis=self.axis, copy=copy)

    def _more_tags(self):
        return {"stateless": True}

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.layers.forward(X)

    def loss(self, Out, Targets):
        #return nn.functional.cross_entropy(Out, Targets)
        return nn.functional.binary_cross_entropy_with_logits(Out, Targets)

def average_smoothing(records, past):      #records - twodimensional array (days, nr_of_bucket), past - how many buckets we look in the past
  #  rolled_records = np.roll(records, past, axis = 1)
   # rolled_records = rolled_records[:,past:]
    rolled_records=records[:,:-past] # does the same as above
    result = np.zeros(shape = rolled_records.shape)
    for day in range(rolled_records.shape[0]):
        for buck in range(rolled_records.shape[1]):
            count_elem = 0.0
            sum_elem = 0.0
            # loop for changing nans to zeros, np.sum(arr) divided by number of nonzero elements
            for elem in records[day][buck:buck+past]:
                if not math.isnan(elem):
                    sum_elem+=elem
                    count_elem+=1
            if count_elem>0:
                result[day][buck] = sum_elem/count_elem
            else:
                result[day][buck] = 0
    return result

def up_still_down(x, epsilon = 10e-7):
    if x>epsilon:
        return 1
    elif x<-epsilon:
        return -1
    else:
        return 0
def test_high(x, epsilon = 0.01):
    if x>epsilon:
        return 1
    else:
        return 0
def test_down(x, epsilon = 0.01):
    if x<epsilon:
        return 1
    else:
        return 0
def up_down_bool(x):
    if x>0:
        return True
    else:
        return False
#np.vectorize(up_still_down)(price_change)

def get_y_to_test_max_bool(true_prices, period_of_getting_max, interval_in_data, threshold=1e-3):
    next_indices = period_of_getting_max//interval_in_data
    num_of_days = true_prices.shape[0]
    num_of_averaged_minutes = true_prices.shape[1]
    res = np.empty(true_prices.shape)
    for i in range(num_of_days):
        for j in range(num_of_averaged_minutes):
            max_val=np.max(true_prices[i][j:j+next_indices])
            true_price=true_prices[i,j]
            if true_price==0:
                true_price=prev_true_price
                print("true price at index {},{} was zero!".format(i,j))
            else:
                prev_true_price=true_prices[i,j]
            if (max_val-true_price)/true_price>=threshold: # price swing >= than threshold % of price
                res[i][j] = True
            else:
                res[i,j]=False
    return res
def get_y_to_test_min_bool(true_prices, period_of_getting_min, interval_in_data, threshold=1e-3):
    next_indices = period_of_getting_min//interval_in_data
    num_of_days = true_prices.shape[0]
    num_of_averaged_minutes = true_prices.shape[1]
    res = np.empty(true_prices.shape)
    for i in range(num_of_days):
        for j in range(num_of_averaged_minutes):
            min_val=np.min(true_prices[i][j:j+next_indices])
            true_price=true_prices[i,j]
            if true_price==0:
                true_price=prev_true_price
                print("true price at index {},{} was zero!".format(i,j))
            else:
                prev_true_price=true_prices[i,j]
            if (min_val-true_price)/true_price<=-threshold: # price swing >= than threshold % of price
                res[i][j] = True
            else:
                res[i,j]=False

    return res

functions_to_call = {
    'mid_price'                              : get_mid_price_comp,
    'true_price'                             : get_true_price_comp,
    'order_inbalance'                        : get_order_inbalance_comp,
    'vwaps_buy'                              : get_vwap_and_ordersizes_comp,
    'vwaps_sell'                             : get_vwap_and_ordersizes_comp,
    'vwaps_order_sizes_buy'                  : get_vwap_and_ordersizes_comp,
    'vwaps_order_sizes_sell'                 : get_vwap_and_ordersizes_comp,
    's2f_impact_buy'                         : get_s2f_impact_and_ordersizes_comp,
    's2f_impact_sell'                        : get_s2f_impact_and_ordersizes_comp,
    's2f_order_sizes_buy'                    : get_s2f_impact_and_ordersizes_comp,
    's2f_order_sizes_sell'                   : get_s2f_impact_and_ordersizes_comp,
    'trading_volumes'                        : get_trading_volume_and_price_volatility,
    'price_volatilities'                     : get_trading_volume_and_price_volatility,
    'next_trade_time'                        : get_next_trade_x_comp,
    'next_trade_size'                        : get_next_trade_x_comp,
    'next_trade_price'                       : get_next_trade_x_comp,
    'price_change'                           : get_price_change_given_prices
}

def get_X(comp_id=1, interval=5, time_to_skip=0, time_back=1,     names=[],     flat_out=True,  ): # use up_and_down or not
    if flat_out:
        X = np.empty(shape=(-10*time_back+10*((510-time_to_skip)//interval), len(names)), dtype=np.float32)
    else:
        X = np.empty(shape=(10, len(names), -time_back+((510-time_to_skip)//interval)))

    #average_smoothing reduces shape "time_back" times per day
    index=0
    s2f_called=False
    vwap_called=False
    volvol_called=False #volume and price volatility
    s2f_imp=np.empty(shape=(10, (510-time_to_skip)//interval))
    s2f_ord=np.empty(shape=(10, (510-time_to_skip)//interval))
    vwap=np.empty(shape=(10, (510-time_to_skip)//interval))
    vwap_ord=np.empty(shape=(10, (510-time_to_skip)//interval))
    volume=np.empty(shape=(10, (510-time_to_skip)//interval))
    volatility=np.empty(shape=(10, (510-time_to_skip)//interval))
    for name in names:
        if name in ['mid_price', 'true_price', 'order_inbalance']:
            arr = functions_to_call[name](comp_id, interval, time_to_skip)
        elif re.search("vwap", name):
            if not vwap_called:
                vwap, vwap_ord=functions_to_call[name](comp_id, interval, time_to_skip)
                vwap_called=True
            if name in ['vwaps_buy']:
                arr = vwap[:,::2]
            elif name in ['vwaps_sell']:
                arr = vwap[:,1::2]
            elif name in ['vwaps_order_sizes_buy']:
                arr = vwap_ord[:,::2]
            elif name in ['vwaps_order_sizes_sell']:
                arr = vwap_ord[:,1::2]
        elif re.search("s2f", name):
            if not s2f_called:
                s2f_imp, s2f_ord=functions_to_call[name](comp_id, interval, time_to_skip)
                s2f_called=True
            if name in ['s2f_impact_buy']:
                arr = s2f_imp[:,::2]
            elif name in ['s2f_impact_sell']:
                arr = s2f_imp[:,1::2]
            elif name in ['s2f_order_sizes_buy']:
                arr = s2f_ord[:,::2]
            elif name in ['s2f_order_sizes_sell']:
                arr = s2f_ord[:,1::2]
        elif name in ['trading_volumes', 'price_volatilities']:
            if not volvol_called:
                volume, volatility=functions_to_call[name](comp_id, interval, time_to_skip)
                volvol_called=True
            if name == 'trading_volumes':
                arr = volume
            elif name == 'price_volatilities':
                arr = volatility
        elif name == 'next_trade_time':
            arr = functions_to_call[name](comp_id, interval, time_to_skip, get_x='time')
        elif name == 'next_trade_size':
            arr = functions_to_call[name](comp_id, interval, time_to_skip, get_x='size')
        elif name == 'next_trade_price':
            arr = functions_to_call[name](comp_id, interval, time_to_skip, get_x='price')
        if flat_out:
            X[:,index]=np.asarray(average_smoothing(arr, time_back).flatten(), dtype=np.float32)
        else:
            pass
        index+=1
    return X

def get_y(comp_id=1, interval=5, time_to_skip=0, time_back=1, y_name='true_price', check='change',swing_interval=30, threshold=1e-2):
    y = functions_to_call[y_name](comp_id, interval, time_to_skip)
    if check=='change':
        y=functions_to_call['price_change'](y)
      #  y=np.vectorize(up_still_down)(y,epsilon*average_comp_price[comp_id])
    elif check=='swing_max':
        y=get_y_to_test_max_bool(y, swing_interval, interval, threshold)
    elif check=='swing_min':
        y=get_y_to_test_min_bool(y, swing_interval, interval, threshold)
    else:
        raise ValueError('bad check argument')
    y = np.asarray(y[:,time_back:].flatten(), dtype=np.float32)
    return y

def logistic_regr(X, y, split_percent, comp_id, epsilon=2.7e-6,
                               use_pipe=preprocessing.MaxAbsScaler(),
                               check_up_down=True,
                               local_names=[]],
                               track_params=False,
                               param_dict_logreg={}}):
    if check_up_down:
        y=np.vectorize(up_still_down)(y,epsilon*average_comp_price[comp_id])#currently we hold price change in y
    split=int(y.size*split_percent)
    if use_pipe:
        clf=make_pipeline(use_pipe, LogisticRegression(max_iter=65 ,solver='sag', tol=1e-3)).fit(X[:split], y[:split])
        if track_params:
            for i in range(len(local_names)):
                param_dict_logreg[local_names[i]].append( clf.named_steps['logisticregression'].coef_[0,i])
        else:
            print("Coefficient impact:")
            for i in range(len(local_names)):
                print(local_names[i], " ", clf.named_steps['logisticregression'].coef_[0,i])
    else:
        clf = LogisticRegression(max_iter=65 ,solver='sag', tol=1e-3).fit(X[:split], y[:split])
        if track_params:
            for i in range(len(local_names)):
                param_dict_logreg[local_names[i]].append( clf.coef_[0,i])
        else:
            print("Coefficient impact:")
            for i in range(len(local_names)):
                print(local_names[i], " ", clf.coef_[0,i])
    y_predicted=clf.predict(X[split:])

    conf_mat=my_confusion_matrix(y[split:], y_predicted, labels=[1,0])
    print(conf_mat)
    precision=conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
    recall=conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0])

    return clf.score(X[split:], y[split:]), precision, recall


def decision_tree(X, y, split_percent, comp_id, epsilon=2.7e-6,
                               use_pipe=preprocessing.MaxAbsScaler(),
                               check_up_down=True,
                               local_names=[]],
                               track_params=False):
    if check_up_down:
        y=np.vectorize(up_still_down)(y,epsilon*average_comp_price[comp_id])#currently we hold price change in y
    split=int(y.size*split_percent)
    if use_pipe:
        clf=make_pipeline(use_pipe, DecisionTreeClassifier(max_depth=13)).fit(X[:split], y[:split])
        if track_params:
            for i in range(len(local_names)):
                param_dict_logreg[local_names[i]].append( clf.named_steps['decisiontreeclassifier'].feature_importances_[i])
        else:
            print("Coefficient impact:")
            for i in range(len(local_names)):
                print(local_names[i], " ", clf.named_steps['decisiontreeclassifier'].feature_importances_[i])
    else:
        clf = DecisionTreeClassifier(max_depth=13).fit(X[:split], y[:split])
        if track_params:
            for i in range(len(local_names)):
                param_dict_logreg[local_names[i]].append( clf.feature_importances_[i])
        else:
            print("Coefficient impact:")
          #  print(clf.feature_importances_)
            for i in range(len(local_names)):
                print(local_names[i], " ", clf.feature_importances_[i])
    y_predicted=clf.predict(X[split:])
    conf_mat=my_confusion_matrix(y[split:], y_predicted, labels=[1,0])
    print(conf_mat)
    precision=conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
    recall=conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0])

    return clf.score(X[split:], y[split:]), precision, recall

def XGB(X, y, split_percent, comp_id, epsilon=2.7e-6,
                               use_pipe=preprocessing.MaxAbsScaler(),
                               check_up_down=False,
                               local_names=[]],
                               track_params=False):
    if check_up_down:
        y=np.vectorize(up_still_down)(y,epsilon*average_comp_price[comp_id])#currently we hold price change in y
    split=int(y.size*split_percent)
    y_train = y[:split]
    neg_class_count = np.sum(y_train==0)
    pos_class_count = np.sum(y_train==1)
    if use_pipe:
        clf=make_pipeline(use_pipe,  xgb.XGBClassifier(max_depth=5, scale_pos_weight=neg_class_count/pos_class_count)).fit(X[:split], y[:split])
      #  print(clf.named_steps)
    else:
        clf =xgb.XGBClassifier().fit(X[:split], y[:split])
    y_predicted=clf.predict(X[split:])
    y_test = y[split:]
    X_test = X[split:]
    conf_mat = confusion_matrix(y_test, y_predicted, labels = [1, 0]).T
    #print(conf_mat)
    if np.unique(y_test).size>1:
        print("Precision:")
        precision = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
        print(precision)
        print("Recall:")
        recall = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
        print(recall)
        return clf.score(X_test, y_test), precision, recall
    else:
        return clf.score(X_test, y_test), math.nan, math.nan


def logistic_regr_cross_val(X_train, y_train, X_test, y_test,
                               use_scaling=True,
                               local_names=names,
                               track_params=False):
    if use_scaling:
        clf=make_pipeline(preprocessing.MaxAbsScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
      #  print(clf.named_steps)
        if track_params:
            for i in range(len(local_names)):
                param_dict_logreg[local_names[i]].append( clf.named_steps['logisticregression'].coef_[0,i])
        else:
            print("Coefficient impact:")
            for i in range(len(local_names)):
                print(local_names[i], " ", clf.named_steps['logisticregression'].coef_[0,i])
    else:
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        if track_params:
            for i in range(len(local_names)):
                param_dict_logreg[local_names[i]].append( clf.coef_[0,i])
        else:
            print("Coefficient impact:")
            for i in range(len(local_names)):
                print(local_names[i], " ", clf.coef_[0,i])
    y_predicted=clf.predict(X_test)
    print("Number of positive class in training dataset =", np.sum(y_train==1))
    print("Number of negative class in training dataset =", np.sum(y_train==0))
    print("Number of positive class in testing dataset =", np.sum(y_test==1))
    print("Number of negative class in testing dataset =", np.sum(y_test==0))
    print("Percent of correct classification:")
    print(np.sum(y_predicted == y_test)/len(y_predicted))
    print("Confusion matrix:")
    conf_mat = confusion_matrix(y_test, y_predicted, labels = [1, 0]).T
    print(conf_mat)
    if np.unique(y_test).size>1:
        print("Precision:")
        precision = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
        print(precision)
        print("Recall:")
        recall = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
        print(recall)
        return clf.score(X_test, y_test), precision, recall
    else:
        return clf.score(X_test, y_test), math.nan, math.nan

def xgb_cross_val(X_train, y_train, X_test, y_test, use_scaling=True):
    neg_class_count = np.sum(y_train==0)
    pos_class_count = np.sum(y_train==1)
    if use_scaling:
        clf=make_pipeline(preprocessing.MaxAbsScaler(), xgb.XGBClassifier(max_depth=5, scale_pos_weight=neg_class_count/pos_class_count)).fit(X_train, y_train)
    else:
        clf = xgb.XGBClassifier().fit(X_train, y_train)
    y_predicted=clf.predict(X_test)
    print("Number of positive class in training dataset =", np.sum(y_train==1))
    print("Number of negative class in training dataset =", np.sum(y_train==0))
    print("Number of positive class in testing dataset =", np.sum(y_test==1))
    print("Number of negative class in testing dataset =", np.sum(y_test==0))
    print("Percent of correct classification:")
    print(np.sum(y_predicted == y_test)/len(y_predicted))
    print("Confusion matrix:")
    conf_mat = confusion_matrix(y_test, y_predicted, labels = [1, 0]).T
    print(conf_mat)
    if np.unique(y_test).size>1:
        print("Precision:")
        precision = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
        print(precision)
        print("Recall:")
        recall = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
        print(recall)
    #print("mse: ", mse(y[split:], y_predicted))
    #print("logistic regr score: ", clf.score(X[split:], y[split:]))
        return clf.score(X_test, y_test), precision, recall
    else:
        return clf.score(X_test, y_test), math.nan, math.nan

def dec_tree_cross_val(X_train, y_train, X_test, y_test, use_scaling=True):
    neg_class_count = np.sum(y_train==0)
    pos_class_count = np.sum(y_train==1)
    if use_scaling:
        clf=make_pipeline(preprocessing.MaxAbsScaler(), DecisionTreeClassifier()).fit(X_train, y_train)
    else:
        clf = DecisionTreeClassifier().fit(X_train, y_train)
    y_predicted=clf.predict(X_test)
    print("Number of positive class in training dataset =", np.sum(y_train==1))
    print("Number of negative class in training dataset =", np.sum(y_train==0))
    print("Number of positive class in testing dataset =", np.sum(y_test==1))
    print("Number of negative class in testing dataset =", np.sum(y_test==0))
    print("Percent of correct classification:")
    print(np.sum(y_predicted == y_test)/len(y_predicted))
    print("Confusion matrix:")
    conf_mat = confusion_matrix(y_test, y_predicted, labels = [1, 0]).T
    print(conf_mat)
    if np.unique(y_test).size>1:
        print("Precision:")
        precision = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
        print(precision)
        print("Recall:")
        recall = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
        print(recall)
    #print("mse: ", mse(y[split:], y_predicted))
    #print("logistic regr score: ", clf.score(X[split:], y[split:]))
        return clf.score(X_test, y_test), precision, recall
    else:
        return clf.score(X_test, y_test), math.nan, math.nan

def cross_validation(X, y, model, num_of_day = 10, use_scaling = True):
    num_of_samples_in_day = X.shape[0]//num_of_day
    plot_indicators = np.zeros(shape = (10, 3))
    for day in range(num_of_day):
        try:
            X_train = np.concatenate((X[:day*num_of_samples_in_day], X[(day+1)*num_of_samples_in_day:]))
            y_train = np.concatenate((y[:day*num_of_samples_in_day], y[(day+1)*num_of_samples_in_day:]))
            X_test = X[day*num_of_samples_in_day:(day+1)*num_of_samples_in_day]
            y_test = y[day*num_of_samples_in_day:(day+1)*num_of_samples_in_day]
            if model == 'log_reg':
                score, precision, recall = logistic_regr_cross_val(X_train, y_train, X_test, y_test, use_scaling = use_scaling, local_names=names,track_params=False)
                plot_indicators[day][0] = score
                plot_indicators[day][1] = precision
                plot_indicators[day][2] = recall
            elif model == 'xgb':
                score, precision, recall = xgb_cross_val(X_train, y_train, X_test, y_test, use_scaling = use_scaling)
                plot_indicators[day][0] = score
                plot_indicators[day][1] = precision
                plot_indicators[day][2] = recall
            elif model == 'dec_tree':
                score, precision, recall = dec_tree_cross_val(X_train, y_train, X_test, y_test, use_scaling = use_scaling)
                plot_indicators[day][0] = score
                plot_indicators[day][1] = precision
                plot_indicators[day][2] = recall
            elif model == 'MLP':
                score, precision, recall = MLP_cross_val(X_train, y_train, X_test, y_test, use_scaling = use_scaling)
                plot_indicators[day][0] = score
                plot_indicators[day][1] = precision
                plot_indicators[day][2] = recall
            elif model == 'LSTM':
                score, precision, recall = LSTM_cross_val(X_train, y_train, X_test, y_test, use_scaling = use_scaling)
                plot_indicators[day][0] = score
                plot_indicators[day][1] = precision
                plot_indicators[day][2] = recall
        except ValueError as err:
            plot_indicators[day][0] = math.nan
            plot_indicators[day][1] = math.nan
            plot_indicators[day][2] = math.nan
            print(err)
    return plot_indicators

def permutate_Xy(X, y):
    assert(len(X.shape)==2)
    s1, s2=X.shape
    assert(y.size==s1)
    Xy=np.empty(shape=(s1,s2+1))
    Xy[:,:s2]=X
    Xy[:,s2]=y
    Xy=np.random.permutation(Xy)
    return Xy[:,:s2], Xy[:,s2]

def my_confusion_matrix(y_true, y_pred, labels): # for binary clasification
    if len(labels)!=2:
        raise ValueError("wrong labels fro binary clasiffication")
    if len(y_true)!=len(y_pred):
        raise ValueError("shapes for y differ")
    truth=labels[0]
    tp=0
    fp=0 #truth was false, prediction was true
    fn=0 #truth was true, prediction was false
    tn=0
    for i in range(len(y_true)):
        if y_true[i]==truth: # true positive or false negative
            if y_true[i]==y_pred[i]:
                tp+=1
            else:
                fn+=1
        else: # true negative or false positive
            if y_true[i]==y_pred[i]:
                tn+=1
            else:
                fp+=1
    return np.array([[tp, fp],[fn, tn]])
def get_model_precision_recall(model, X, Y, device="cpu"):
    model.eval()
    model.to(device)
    conf_m=np.zeros((2,2))
    with torch.no_grad():
        #for x, y in zip(X,Y):
        x = X.to(device)
        y = Y.to(device)
        outputs = model.forward(x)
        if len(y.shape)==1:
          _, predictions = outputs.data.max(dim=1)
          conf_m=(my_confusion_matrix(y, predictions, labels=[1,0]))
          print(conf_m)
          print("precision: {}, recall: {}".format(conf_m[0,0]/(conf_m[0,0]+conf_m[0,1]), conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])))
        else:
          _, predictions = outputs.data.max(dim=1)
          _, y =y.data.max(dim=1)
          conf_m=(my_confusion_matrix(y, predictions, labels=[1,0]))
          print(conf_m)
          print("precision: {}, recall: {}".format(conf_m[0,0]/(conf_m[0,0]+conf_m[0,1]), conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])))
    return conf_m[0,0]/(conf_m[0,0]+conf_m[0,1]), conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])

def compute_error_rate(model, X, Y, device="cpu"):

    model.eval()
    model.to(device)


    num_errs = 0.0
    num_examples = 0

    with torch.no_grad():
        #for x, y in zip(X,Y):
        x = X.to(device)
        y = Y.to(device)
        outputs = model.forward(x)
        _, predictions = outputs.data.max(dim=1)
        if len(y.shape)!=1:
          _, y =y.data.max(dim=1)
          num_errs += (predictions != y).sum().item()
        else:
          num_errs += (predictions != y.data).sum().item()
        num_examples += x.size(0)
    return num_errs / num_examples
def plot_history(history):
    """Helper to plot the trainig progress over time."""
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.grid()
    train_loss = np.array(history["train_losses"])
    plt.semilogy(np.arange(train_loss.shape[0]), train_loss, label="batch train loss")
    val_loss = np.array(history["val_losses"])
    plt.semilogy(np.arange(val_loss.shape[0]), val_loss, label="val loss")
    #plt.ylim(0, 0.20)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.grid()
    train_errs = np.array(history["train_errs"])
    plt.plot(np.arange(train_errs.shape[0]), train_errs, label="batch train error rate")
    val_errs = np.array(history["val_errs"])
    plt.plot(val_errs[:, 0], val_errs[:, 1], label="validation error rate", color="r")
    #plt.ylim(0, 0.20)
    plt.legend()

def SGD(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    alpha=1e-4,
    momentum=0.1,
    decay=0.01,
    beta=0.95,
    batch_size=100,
    num_epochs=3,
    max_num_epochs=np.nan,
    patience_expansion=1.5,
    log_every=100,
    polyak=False,
    device="cpu",
):

    # Put the model in train mode, and move to the evaluation device.
    model.train()
    model.to(device)
    X_valid=X_valid.to(device)
    y_valid=y_valid.to(device)
    velocities = [torch.zeros_like(m) for m in model.parameters()]
   # if polyak:
    polyaks= [torch.zeros_like(m) for m in model.parameters()]
    #print(velocities)
    #
    alpha0=alpha
    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "val_losses": [], "train_errs": [], "val_errs": []}
    print("Training the model!")
    print("Interrupt at any time to evaluate the best validation model so far.")
 #   try:
    tstart = time.time()
    siter = iter_
    while epoch < num_epochs:
        model.train()
        epoch += 1
        if epoch > max_num_epochs:
            break
        alpha=alpha0*np.power(beta,epoch)

        data_permutation=torch.randperm(X_train.size(0))
        X_train=X_train[data_permutation]
        y_train=y_train[data_permutation]
      #  counters=np.random.permutation(np.arange(X_train.size(0)//batch_size))
        for counter in range(X_train.size(0)//batch_size):#zip(X_train,y_train):
            x=X_train[counter*batch_size:(counter+1)*batch_size]
            x = x.to(device)
            y=y_train[counter*batch_size:(counter+1)*batch_size]
            y = y.to(device)
            iter_ += 1
            out = model(x)
            loss = model.loss(out, y)
            loss.backward()
            _, predictions = out.max(dim=1)
            if len(y.shape)>1:
              _,y=y.data.max(dim=1)
            batch_err_rate = (predictions != y).sum().item() / out.size(0)

            history["train_losses"].append(loss.item())
            history["train_errs"].append(batch_err_rate)

            with torch.no_grad():
                for (name, p), v, pol in zip(model.named_parameters(), velocities, polyaks):
                    if "weight" in name:
                        p.grad +=  decay * 2*p

                    v[:] = momentum*v - alpha * p.grad
                    p[:] =p + v
                    pol[:]=0.99*pol + (1-0.99)*p
                    p.grad.zero_()

            if iter_ % log_every == 0:
                num_iter = iter_ - siter + 1
                print(
                    "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%, steps/s {3: >5.2f}".format(
                        iter_,
                        loss.item(),
                        batch_err_rate * 100.0,
                        num_iter / (time.time() - tstart),
                    )
                )
                tstart = time.time()

        val_err_rate = compute_error_rate(model, X_valid, y_valid, device)
        history["val_errs"].append((iter_, val_err_rate))

        if val_err_rate < best_val_err:
            # Adjust num of epochs
            num_epochs = int(np.maximum(num_epochs, epoch * patience_expansion + 1))
            best_epoch = epoch
            best_val_err = val_err_rate
            best_params = [p.detach().cpu() for p in model.parameters()]
        clear_output(True)
        m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs".format(
            epoch, val_err_rate * 100.0, num_epochs
        )
        print("{0}\n{1}\n{0}".format("-" * len(m), m))

        if(polyak):
            with torch.no_grad():
                for (name, p), pol in zip(model.named_parameters(), polyaks):
                    p[:]=pol
                    pol[:]=torch.zeros_like(pol)


    #except KeyboardInterrupt:
     # pass

    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
    plot_history(history)
    return history


def Adam(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    alpha=1e-4,
    momentum=0.1,
    decay=0.01,
    beta=0.95,
    batch_size=100,
    num_epochs=3,
    max_num_epochs=np.nan,
    patience_expansion=1.5,
    log_every=100,
    decrease_alpha_every=10000,
    polyak=False,
    device="cpu",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=1e-5)
    # Put the model in train mode, and move to the evaluation device.
    model.train()
    model.to(device)
    X_valid=X_valid.to(device)
    y_valid=y_valid.to(device)
    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "val_losses": [], "train_errs": [], "val_errs": []}
    print("Training the model!")
    print("Interrupt at any time to evaluate the best validation model so far.")
    try:
        tstart = time.time()
        siter = iter_
        while epoch < num_epochs:
            model.train()
            epoch += 1
            if epoch > max_num_epochs:
                break

            data_permutation=torch.randperm(X_train.size(0))
            X_train=X_train[data_permutation]
            y_train=y_train[data_permutation]
          #  counters=np.random.permutation(np.arange(X_train.size(0)//batch_size))
            for counter in range(X_train.size(0)//batch_size):#zip(X_train,y_train):
                x=X_train[counter*batch_size:(counter+1)*batch_size]
                x = x.to(device)
                y=y_train[counter*batch_size:(counter+1)*batch_size]
                y = y.to(device)#.reshape(1)
                iter_ += 1
                out = model(x)
                loss = model.loss(out, y)
                _, predictions = out.max(dim=1)
                if len(y.shape)>1:
                  _,y=y.data.max(dim=1)
                batch_err_rate = (predictions != y).sum().item() / out.size(0)

                history["train_losses"].append(loss.item())
                history["train_errs"].append(batch_err_rate)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_ % log_every == 0:
                    num_iter = iter_ - siter + 1
                    print(
                        "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%, steps/s {3: >5.2f}".format(
                            iter_,
                            loss.item(),
                            batch_err_rate * 100.0,
                            num_iter / (time.time() - tstart),
                        )
                    )
                    tstart = time.time()

            val_err_rate = compute_error_rate(model, X_valid, y_valid, device)
            history["val_errs"].append((iter_, val_err_rate))

            if val_err_rate < best_val_err:
                # Adjust num of epochs
                num_epochs = int(np.maximum(num_epochs, epoch * patience_expansion + 1))
                best_epoch = epoch
                best_val_err = val_err_rate
                best_params = [p.detach().cpu() for p in model.parameters()]
            clear_output(True)
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs".format(
                epoch, val_err_rate * 100.0, num_epochs
            )
            print("{0}\n{1}\n{0}".format("-" * len(m), m))



    except KeyboardInterrupt:
        pass

    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
    plot_history(history)
    return history

def prep_y_MLP(y):# 1->[0,1],0->[1,0]
  y_out=torch.zeros(size=(y.shape[0], 2))
  for i in range(y.shape[0]):
    if y[i]==0:
      y_out[i]=torch.tensor([1,0])
    else:
      y_out[i]=torch.tensor([0,1])
  return y_out

def MLP_wrapper(X, y, names):
  X_comp=torch.tensor(X, dtype=torch.float32)
  y_comp=torch.tensor(y, dtype=torch.long)
  split=int(X_all.shape[1]*0.6)
  valid=int(X_all.shape[1]*0.8)

  if len(X_comp.shape)==1:
    X_comp=torch.atleast_2d(X_comp).T

  Xy__comp_train=np.empty(shape=(X_comp[:split].shape[0], X_comp[:split].shape[1]+1))
  Xy__comp_train[:,: X_comp.shape[1]]=X_comp[:split]
  Xy__comp_train[:,X_comp.shape[1]]=y_comp[:split]
  Xy__comp_train=np.random.permutation(Xy__comp_train)
  X_comp_train=Xy__comp_train[:,:X_comp.shape[1]]

  X_comp_train=torch.tensor(X_comp_train, dtype=torch.float32)


  y_comp_train=prep_y_MLP(Xy__comp_train[:,X_comp.shape[1]])

  y_comp_valid=prep_y_MLP(y_comp[split:valid])


  MLP=Model(nn.Linear(len(names), 128), nn.Sigmoid(), nn.Dropout(p=0.2),
          nn.Linear(128, 512), nn.Sigmoid(),        nn.Dropout(p=0.5),
          nn.Linear(512, 1024), nn.Sigmoid(),       nn.Dropout(p=0.5),
          nn.Linear(1024, 4096), nn.Sigmoid(),      nn.Dropout(p=0.5),
          nn.Linear(4096, 4096), nn.Sigmoid(),      nn.Dropout(p=0.5),
       #   nn.Linear(4096, 4096), nn.Sigmoid(),      #nn.Dropout(p=0.5),
          nn.Linear(4096,1024), nn.Sigmoid(),       nn.Dropout(p=0.5),
          nn.Linear(1024,64), nn.Sigmoid(),         nn.Dropout(p=0.2),
          nn.Linear(64,2),# nn.ReLU(),
          #nn.Linear(8,2)
         )
  with torch.no_grad():
      # Initialize parameters
      for name, p in MLP.named_parameters():
          if "weight" in name:
              p.normal_(1e-8, 0.5)
          elif "bias" in name:
              #p.zero_()
              p.normal_(0, 1e-4)
          else:
              raise ValueError('Unknown parameter name "%s"' % name)
  t_start = time.time()

  SGD(MLP, X_comp_train, y_comp_train,
      X_comp[split:valid],y_comp_valid,
      alpha=1e-2, momentum=1e-5, decay=1e-5,beta=0.99,batch_size=1024, num_epochs=100, max_num_epochs=200, device='cuda')#, polyak=True)



  test_err_rate = compute_error_rate(MLP,
                                      X_comp[valid:],y_comp[valid:], device='cuda',
                                    )
  precision, recall=get_model_precision_recall(MLP,   X_comp[valid:],y_comp[valid:], device='cuda')


  return 1-test_err_rate, precision, recall
