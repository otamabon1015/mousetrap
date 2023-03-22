"""
@project: BA-Thesis: The best ways to catch a mouse without a mouse trap: Bridging the gap between behavioral research and machine learning methods
@author: Tamaki Ogura (ogura@cl.uni-heidelberg.de)
@filename: test.py
@description: contain functions for evaluation of trained ML and DL models
"""

import glob
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)


def create_performance_df(acc, balanced_acc, f1):
    """
    Create dataframe containing accuracy, balanced accuracy and f1 score.

    Args: 
        acc, balanced_acc, f1: scores calculated for each trained model
    
    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    performance_dict = {
        'accuracy':[],
        'balanced_acc':[],
        'f1score':[],
    }
    
    performance_dict['accuracy'].append(round(acc, 4))
    performance_dict['balanced_acc'].append(round(balanced_acc, 4))
    performance_dict['f1score'].append(round(f1, 4))


    performance_df = pd.DataFrame.from_dict(performance_dict)

    return performance_df

def eval_ml_model_(model, X_test):
    """ 
    Evaluate trained ML models on a test set.

    Args: 
        model: trained ML model
        X_test: test set which the model will be evaluated on
    
    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    X_test, y_test = X_test.drop(["Condition", "subject_nr"], axis=1), X_test["Condition"]
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(" Accuracy: {:.4f} \n Balanced Accuracy: {:.4f} \n F1 Score: {:.4f} ".format(acc, balanced_acc, f1))
    
    print(' Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\n')
    
    df = create_performance_df(acc, balanced_acc, f1)
    
    return df

def eval_ml_model(model, X_test1, X_test2):
    """ 
    Evaluate trained ML models on both test sets.

    Args: 
        model: trained ML model
        X_test1, X_test2: test set which the model will be evaluated on

    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    print('\nResult on Stroop test set...')
    result_stroop = eval_ml_model_(model, X_test1)

    print('\nResult on Seminar test set...')
    result_seminar = eval_ml_model_(model, X_test2)

    print(result_stroop)
    print(result_seminar)
    
    return result_stroop, result_seminar


def eval_dl_model_(model, test_loader):
    """ 
    Evaluate trained DL models on a test set.

    Args: 
        model: trained DL model
        test_loader: test set which the model will be evaluated on
    
    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    y_test, y_pred = [], []

    for x, y in test_loader:
        out = model(x)
        _, pred = torch.max(out, 1)
        y_test.append(np.asarray(y))
        y_pred.append(np.asarray(pred))
    
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(" Accuracy: {:.4f} \n Balanced Accuracy \n F1 Score: {:.4f} : {:.4f}".format(acc, balanced_acc, f1))
    
    print(' Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\n')
    
    df = create_performance_df(acc, balanced_acc, f1)
    
    return df

def eval_dl_model(model, test_loader1, test_loader2):
    """ 
    Evaluate trained DL models on both test sets.

    Args: 
        model: trained DL model
        test_loader1, test_loader2: test set which the model will be evaluated on
    
    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    print('\nResult on Stroop test set...')
    result_stroop = eval_dl_model_(model, test_loader1)

    print('\nResult on Seminar test set...')
    result_seminar = eval_dl_model_(model, test_loader2)
    
    return result_stroop, result_seminar


def eval_gcn_model_(model, test_loader):
    """ 
    Evaluate trained GCN model on a test set.

    Args: 
        model: trained DL model
        test_loader: test set which the model will be evaluated on
    
    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    y_test, y_pred = [], []

    for data in test_loader:
        out = model(data)
        _, pred = out.max(dim=1)
        y_test.append(data.y.cpu())
        y_pred.append(pred.cpu())
        
    y_test, y_pred = torch.cat(y_test, dim=0).numpy(), torch.cat(y_pred, dim=0).numpy()

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(" Accuracy: {:.4f} \n Balanced Accuracy \n F1 Score: {:.4f} : {:.4f}".format(acc, balanced_acc, f1))
    
    print(' Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\n')

    df = create_performance_df(acc, balanced_acc, f1)
    
    return df

def eval_gcn_model(model, test_loader1, test_loader2):
    """ 
    Evaluate trained GCN on both test sets.

    Args: 
        model: trained DL model
        test_loader1, test_loader2: test set which the model will be evaluated on
    
    Return:
        performance_df: dataframe that contains the result of testing the model
    """
    print('\nResult on Stroop test set...')
    result_stroop = eval_gcn_model_(model, test_loader1)

    print('\nResult on Seminar test set...')
    result_seminar = eval_gcn_model_(model, test_loader2)
    
    return result_stroop, result_seminar

def get_loaded_model(model_type):
    """ 
    Load the trained ML or DL models.

    Args: 
        model_type: ML or DL models
    
    Return:
        trained_models_list: list containing the trained ML or DL models
    """
    trained_models_path = f'./trained_models/{model_type}/*'
    trained_models_list = glob.glob(trained_models_path)

    return trained_models_list

    