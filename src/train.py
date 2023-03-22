"""
@project: BA-Thesis: The best ways to catch a mouse without a mouse trap: Bridging the gap between behavioral research and machine learning methods
@author: Tamaki Ogura (ogura@cl.uni-heidelberg.de)
@filename: train.py
@description: contain functions for training ML and DL models
"""

import logging
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)


def LOOCV(model, data, ids, outcomevar, dropcols, idcolumn):
    """
    Implementation for "Leave-One-Participant-Out" Cross Validation.
    The entire code was programmed based on https://github.com/Big-Ideas-Lab/DBDP/tree/master/DigitalBiomarkers-generalML/loocvRF 

    Args:
        model: initialized model that will be trained
        data: dataframe containing training data ("Condition" and "subject_nr" columns should be included)
        ids: participant's number ("subject_nr") held out for validation 
        outcomevar: participant's number used for training
        dropcols: columns that will be droped
        idcolumn: column containing participant's number
    """
    LOOCV_O = str(ids)
    data[idcolumn] = data[idcolumn].apply(str)
    data_filtered = data[data[idcolumn] != LOOCV_O]
    data_cv = data[data[idcolumn] == LOOCV_O]
   
    # Test data - the person left out of training
    data_test = data_cv.drop(columns=dropcols)
    X_test = data_test.drop(columns=[outcomevar])
    y_test = data_test[outcomevar] #This is the outcome variable
    
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=dropcols)
    X_train = data_train.drop(columns=[outcomevar])
    
    feature_list = list(X_train.columns)
    X_train = np.array(X_train)
    y_train = np.array(data_train[outcomevar]) #Outcome variable here

    model.fit(X_train, y_train)
    

def train_ml(model, data, idcolumn, outcomevar, dropcols=[]):
    """
        Main function to train ML model training. 
        Args:
          model: initialized model that will be trained
          data (pandas DataFrame): This is a dataframe containing each participant's features and outcome variables
          idcolumn (string): This is the column name of your column containing your participant number or ID (case sensitive)
          outcomevar (string): This is the column name of your outcome variable (case sensitive)
          dropcols (list): This is a list containing strings of each column you wish to drop in your dataframe. Default is empty list [].
    """
    
    # Make list of all ID's in idcolumn
    IDlist = list(set(data["subject_nr"]))
    drop = [idcolumn] #add idcolumn to dropcols to drop from model
    drop = drop + dropcols
    
    # Initialize empty lists and dataframe 
    accuracies = []
    balanced_accuracies = []
    f1scores = []

    logging.info(f'Training {model} started.')
    
    # Run LOOCV! 
    for i in IDlist:
        LOOCV(model, data, i, outcomevar, drop, idcolumn)

    logging.info(f'\n{model} is trained.')

    filename = f'./trained_models/ML/trained_{model}.pkl'
    joblib.dump(model, filename)
    logging.info(f'\nTrained {model} is saved in the "trained_models" folder.')
    
    return

def train_dl(model, train_loader, test_loader, epoches=100, lr=0.001):
    """
    Train DL models except GCN

    Args:
        model: initialized model that will be trained
        train_loader: data loader containing training data
        test_loader: data loader containing test data
        epoches: how many epochs to train
        lr: learning rate
    """
    device = "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model_name = assign_model_name(model)

    logging.info(f'{model_name} will be trained for {epoches} epochs.')
    logging.info(f'Training {model_name} started.')

    for epoch in range(epoches):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader)


        if (epoch+1)%10==0:
            model.eval()
            ys, preds = [], []
            val_loss = 0
            for x, y in test_loader:
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() 
                _, pred = torch.max(out, 1)
                ys.append(np.asarray(y))
                preds.append(np.asarray(pred))

            val_loss /= len(test_loader)
            acc = accuracy_score(ys, preds)
            balanced_acc = balanced_accuracy_score(ys, preds)
            f1 = f1_score(ys, preds)

            print('Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Accuracy: {:.4f}, Balanced Accuracy: {:.4f}, F1 Score: {:.4f}'.format(epoch+1, train_loss, val_loss, acc, balanced_acc, f1))

    logging.info(f'\n{model_name} is trained.')

    filename = f'./trained_models/DL/trained_{model_name}.pt'
    torch.save(model, filename)
    logging.info(f'\nTrained {model_name} is saved in the "trained_models" folder.')
    
    return

def train_gcn(model, train_loader, test_loader, epoches=100, lr=0.001):
    """
    Train GCN model

    Args:
        model: initialized model that will be trained
        train_loader: data loader containing graph-structured training data
        test_loader: data loader containing graph-structured test data
        epoches: how many epochs to train
        lr: learning rate
    """
    device = "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model_name = assign_model_name(model)

    logging.info(f'{model_name} will be trained for {epoches} epochs.')
    logging.info(f'Training {model_name} started.')
        
    for epoch in range(epoches):
        model.train()
        train_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader)


        if (epoch+1)%10==0:
            model.eval()
            ys, preds = [], []
            val_loss = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item() 
                _, pred = out.max(dim=1)
                ys.append(data.y.cpu())
                preds.append(pred.cpu())

            y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
            val_loss /= len(test_loader)
            acc = accuracy_score(y, pred)
            balanced_acc = balanced_accuracy_score(y, pred)
            f1 = f1_score(y, pred)

            print('Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Accuracy: {:.4f}, Balanced Accuracy: {:.4f}, F1 Score: {:.4f}'.format(epoch+1, train_loss, val_loss, acc, balanced_acc, f1))

    logging.info(f'\n{model_name} is trained.')

    filename = f'./trained_models/DL/trained_{model_name}.pt'
    torch.save(model, filename)
    logging.info(f'\nTrained {model_name} is saved in the "trained_models" folder.')


def assign_model_name(model):
    """
    Assign short name for each DL model

    Args: 
        model: initialized model that will be trained
    
    Return:
        model_name: assigned model name 
    """
    if str(model)[0] == 'C':
        model_name = 'cnn'
    elif str(model)[0] == 'G':
        model_name = 'gcn'
    elif str(model)[0] == 'D':
        model_name = 'deepconv_lstm'
    else:
        model_name = 'transformer'
    return model_name
