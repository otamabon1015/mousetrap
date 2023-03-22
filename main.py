"""
@project: BA-Thesis: The best ways to catch a mouse without a mouse trap: Bridging the gap between behavioral research and machine learning methods
@author: Tamaki Ogura (ogura@cl.uni-heidelberg.de)
@filename: main.py
@description: execute the training and testing pipeline
"""

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from src.dl_models import *
from src.prepare_data import *
from src.test import *
from src.train import *

import glob
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier


def setup_logging():
    """
    Setup logging configuration.
    """

    format_prefix = '%(asctime)s - %(levelname)s: %(message)s'
    format_date = '%Y  -%m-%d %H:%M:%S'
    logging.basicConfig(format=format_prefix, level=logging.INFO, datefmt=format_date)
    logging.info('logging is set up.')

def create_folders():
    """
    Create folders to save results.
    """
    trained_models_path = os.path.join('trained_models')
    if not os.path.exists(trained_models_path):
        os.makedirs(trained_models_path)

    results_path = os.path.join('results')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

def import_conf(conf_file = 'conf.json'):
    """
    Import the configurations defined in json file.

    Args:
        conf_file (str): name of the configuration file. 'conf.json' is default.
    Return:
        option: models will be trained or tested on the data.
        balanced: which balanced data to use: undersampled or oversampled data. If 'false', imbalanced data will be used.
        model_type: ML or DL models will be trained.
    """

    with open(conf_file) as json_file:
        conf_dict = json.load(json_file)

    option = conf_dict['general'].get('option')
    model_type = conf_dict['general'].get('model_type')
    balanced = conf_dict['general'].get('balanced')

    if option == 'train':
        logging.info(f'{model_type} models will be trained.')

        if balanced == 'false':
            logging.info('For training imbalanced data will be used.')
        else:
            logging.info(f'For training {balanced} data will be used.')
    elif option == 'test':
        logging.info(f'Trained {model_type} models will be tested.')

    else:
        logging.info('Invalid option.')

    return option, balanced, model_type

def main():
    """
    Execute entire training and testing pipeline.
    """

    # Logging configurations to follow code workflow.
    setup_logging()

    # Create the necessary folders.
    create_folders()

    logging.info('----- Loading configuration file -----')

    # Import the configuration file and the dataset.
    option, balanced, model_type = import_conf()
    
    # Train models
    if option == 'train':

        if model_type == 'ML':

            # Load data
            if balanced == 'false':
                X_train = pd.read_csv('data/Stroop/ML/train_data.csv', sep=',').drop(['Unnamed: 0'], axis=1)
                print(X_train)

            else:
                X_train = pd.read_csv(f'data/Stroop/ML/{balanced}_train_data.csv', sep=',').drop(['Unnamed: 0'], axis=1)
                
            X_test = pd.read_csv('data/Stroop/ML/test_data.csv', sep=',').drop(['Unnamed: 0'], axis=1)
            Seminar_X_test = pd.read_csv(f'data/Seminar/ML/test_data.csv', sep=',').drop(['Unnamed: 0'], axis=1)

            # Scale data
            if balanced == 'false':
                X_train, X_test, Seminar_X_test = scale_data(X_train, X_test, Seminar_X_test)

            # ML models to be trained
            models = [GaussianNB(), LinearDiscriminantAnalysis(), KNeighborsClassifier(), BaggingClassifier(DecisionTreeClassifier()), RandomForestClassifier()]
            results_stroop = []
            results_seminar =[]

            for model in models:
                # Train models using 'leave-one-participant-out' cross validation procedure
                train_ml(model, X_train, "subject_nr", "Condition", dropcols=[])

                # Evaluate the trained model on the test sets 
                result_stroop, result_seminar = eval_ml_model(model, X_test, Seminar_X_test)

                results_stroop.append(result_stroop.iloc[0].tolist())
                results_seminar.append(result_seminar.iloc[0].tolist())
        
            # Save the results as csv file
            results_stroop_df = pd.DataFrame(results_stroop, columns=['accuracy', 'balanced_acc', 'f1score'], index=['NB', 'LDA', 'KNN', 'TB', 'RF'])
            results_stroop_df.to_csv('results/ml_performance_comparison_stroop.csv')
            results_seminar_df = pd.DataFrame(results_seminar, columns=['accuracy', 'balanced_acc', 'f1score'], index=['NB', 'LDA', 'KNN', 'TB', 'RF'])
            results_seminar_df.to_csv('results/ml_performance_comparison_seminar.csv')


        elif model_type == 'DL':

            # Load data
            train_loader, test_loader, seminar_loader = load_data_loader(balanced)

            # DL models to be trained
            # GCN will be trained right after this part, since it requires graph-structured data
            models = [CNN(), DeepConvLSTM(), TransformerClassifier()]
            results_stroop = []
            results_seminar =[]

            for model in models:
                # Train DL models
                train_dl(model, train_loader, test_loader)
                result_stroop, result_seminar = eval_dl_model(model, test_loader, seminar_loader)

                results_stroop.append(result_stroop.iloc[0].tolist())
                results_seminar.append(result_seminar.iloc[0].tolist())

            # Load graph data for GCN
            train_loader, test_loader, seminar_loader = load_graph_loader(balanced)
            model = GCN()
            # Train and evaluate GCN
            train_gcn(model, train_loader, test_loader, epoches=1, lr=0.001)
            result_stroop, result_seminar = eval_gcn_model(model, test_loader, seminar_loader)

            results_stroop.append(result_stroop.iloc[0].tolist())
            results_seminar.append(result_seminar.iloc[0].tolist())

            # Save the results as csv file
            results_stroop_df = pd.DataFrame(results_stroop, columns=['accuracy', 'balanced_acc', 'f1score'], index=['CNN', 'DeepConvLSTM', 'Transformer', 'GCN'])
            results_stroop_df.to_csv('results/dl_performance_comparison_stroop.csv')
            results_seminar_df = pd.DataFrame(results_seminar, columns=['accuracy', 'balanced_acc', 'f1score'], index=['CNN', 'DeepConvLSTM', 'Transformer', 'GCN'])
            results_seminar_df.to_csv('results/dl_performance_comparison_seminar.csv')

        logging.info(f'----- Training of all {model_type} models are finised. -----')
        logging.info(f'----- Results are saved in "results" folder. -----')
    
    # Test models
    elif option == 'test':

        # Load trained ML or DL models
        trained_models_list = get_loaded_model(model_type)
        results_stroop = []
        results_seminar =[]

        if model_type == 'ML':

            # Load test sets
            X_test = pd.read_csv('data/Stroop/ML/test_data.csv', sep=',')
            Seminar_X_test = pd.read_csv(f'data/Seminar/ML/test_data.csv', sep=',')

            # Test ML models and show the results
            for trained_model in trained_models_list:
                logging.info(f'Testing {trained_model}.')
                model = joblib.load(trained_model)
                result_stroop, result_seminar= eval_ml_model(model, X_test, Seminar_X_test)

                results_stroop.append(result_stroop.iloc[0].tolist())
                results_seminar.append(result_seminar.iloc[0].tolist())

            # Save the results as csv file
            results_stroop_df = pd.DataFrame(results_stroop, columns=['accuracy', 'balanced_acc', 'f1score'], index=['CNN', 'DeepConvLSTM', 'Transformer', 'GCN'])
            results_stroop_df.to_csv('results/ml_performance_comparison_stroop.csv')
            results_seminar_df = pd.DataFrame(results_seminar, columns=['accuracy', 'balanced_acc', 'f1score'], index=['CNN', 'DeepConvLSTM', 'Transformer', 'GCN'])
            results_seminar_df.to_csv('results/ml_performance_comparison_seminar.csv')

        elif model_type == 'DL':
            # Load test loaders
            _, test_loader, seminar_loader = load_data_loader(balanced='false')
            
            for trained_model in trained_models_list:

                # If GCN, use graph-structured data instead
                if trained_model[-6:-3] == 'gcn':
                    _, test_loader, seminar_loader = load_graph_loader(balanced='false')
                    logging.info(f'Testing {trained_model}.')
                    model = torch.load(trained_model)
                    result_stroop, result_seminar = eval_gcn_model(model, test_loader, seminar_loader)
                
                # Test other DL models and show the results
                else:
                    logging.info(f'Testing {trained_model}.')
                    model = torch.load(trained_model)
                    result_stroop, result_seminar = eval_dl_model(model, test_loader, seminar_loader)

                results_stroop.append(result_stroop.iloc[0].tolist())
                results_seminar.append(result_seminar.iloc[0].tolist())

            # Save the results as csv file
            results_stroop_df = pd.DataFrame(results_stroop, columns=['accuracy', 'balanced_acc', 'f1score'], index=['CNN', 'DeepConvLSTM', 'Transformer', 'GCN'])
            results_stroop_df.to_csv('results/dl_performance_comparison_stroop.csv')
            results_seminar_df = pd.DataFrame(results_seminar, columns=['accuracy', 'balanced_acc', 'f1score'], index=['CNN', 'DeepConvLSTM', 'Transformer', 'GCN'])
            results_seminar_df.to_csv('results/dl_performance_comparison_seminar.csv')

        logging.info(f'----- Testing of all {model_type} models are finished. -----')

if __name__ == "__main__":
    main()
