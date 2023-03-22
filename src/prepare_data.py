"""
@project: BA-Thesis: The best ways to catch a mouse without a mouse trap: Bridging the gap between behavioral research and machine learning methods
@author: Tamaki Ogura (ogura@cl.uni-heidelberg.de)
@filename: prepare_data.py
@description: prepare data for training and testing models
"""

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, DataLoader


def scale_data(df_train, df_test1, df_test2):
    """
    Scale the training and test data used for training ML models.

    Args:
        df_train: dataframe for training data
        df_test1: dataframe for test data (Stroop)
        df_test2: dataframe for test data (Seminar)
    Return:
        scaled_train_df: scaled dataframe for training data
        scaled_tesy_df1: scaled dataframe for test data (Stroop)
        scaled_test_df2: scaled dataframe for test data (Seminar)
    """
    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(df_train.drop(["subject_nr", "Condition"], axis=1))
    scaled_test1 = scaler.fit_transform(df_test1.drop(["subject_nr", "Condition"], axis=1))
    scaled_test2 = scaler.fit_transform(df_test2.drop(["subject_nr", "Condition"], axis=1))

    scaled_train_df = pd.DataFrame(scaled_train, columns=df_train.drop(["subject_nr", "Condition"], axis=1).columns)
    scaled_train_df = pd.concat([df_train[["subject_nr", "Condition"]], scaled_train_df], axis=1)

    scaled_test_df1 = pd.DataFrame(scaled_test1, columns=df_test1.drop(["subject_nr", "Condition"], axis=1).columns)
    scaled_test_df1 = pd.concat([df_test1[["subject_nr", "Condition"]], scaled_test_df1], axis=1)

    scaled_test_df2 = pd.DataFrame(scaled_test2, columns=df_test2.drop(["subject_nr", "Condition"], axis=1).columns)
    scaled_test_df2 = pd.concat([df_test2[["subject_nr", "Condition"]], scaled_test_df2], axis=1)

    return scaled_train_df, scaled_test_df1, scaled_test_df2

class MTDataset(Dataset):
    """
    Load data for training DL models
    """
    def __init__(self, file_path, dataset, oversampled=False, undersampled=False):
        self.file_path = file_path
        self.dataset = dataset
        if undersampled==True:
            self.label = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/undersampled_label.csv', delimiter=',')
            self.xpos = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/undersampled_xpos.csv', delimiter=',')
            self.ypos = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/undersampled_ypos.csv', delimiter=',')
            
        elif oversampled==True:
            self.label = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/oversampled_label.csv', delimiter=',')
            self.xpos = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/oversampled_xpos.csv', delimiter=',')
            self.ypos = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/oversampled_ypos.csv', delimiter=',')
        
        else:
            self.label = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/label.csv', delimiter=',')
            self.xpos = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/xpos.csv', delimiter=',')
            self.ypos = np.loadtxt('data/' + self.file_path + '/DL/' + self.dataset + '/ypos.csv', delimiter=',')
        
        print(self.file_path + " " + self.dataset, f": {len(self.xpos)}")

        self.data = []
        
        for i in range(len(self.xpos)):
            label = self.label[i]
            xpos = self.xpos[i]
            ypos = self.ypos[i]
            dp = np.array([xpos, ypos])
            self.data.append([dp, label])
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        dp, label = self.data[idx]
        X = torch.tensor(dp).float()
        y = torch.tensor(label).long()
        return X, y

def load_graph(data_loader, file_path, dataset, oversampled=False, undersampled=False):
    """
    Convert data_loader loaded for DL models into graph-structured data.

    Args:
        data_loader: data_loader containing x-pos, y-pos and label
        file_path: which data to use (Stroop, Seminar or Pilot_Study)
        dataset: which dataset to use (train or test)
        oversampled: if True, oversampled data will be loaded
        underampled: if True, undersampled data will be loaded
    Return:
        dataset: graph-structured data consisting of nodes and edges 
    """

    if undersampled==True:
        labels = np.loadtxt('data/' + file_path + '/DL/' + dataset + '/undersampled_label.csv', delimiter=',')
    elif oversampled==True:
        labels = np.loadtxt('data/' + file_path + '/DL/' + dataset + '/oversampled_label.csv', delimiter=',')
    else:
        labels = np.loadtxt('data/' + file_path + '/DL/' + dataset + '/label.csv', delimiter=',')

    count = 0
    dataset = []

    for x, y in data_loader:
        num_vertices = len(x[0][0])
        G = nx.DiGraph()

        for x_pos, y_pos in x:
            for i in range(num_vertices-1):

                G.add_edge(i, i+1)
                if i == num_vertices-2:
                    G.nodes[i+1]["x_pos"] = [float(x_pos[i+1])]
                    G.nodes[i+1]["y_pos"] = [float(y_pos[i+1])]

                G.nodes[i]["x_pos"] = [float(x_pos[i])]
                G.nodes[i]["y_pos"] = [float(y_pos[i])]

            x_pos_list = nx.get_node_attributes(G,'x_pos')
            df_x = pd.DataFrame.from_dict(x_pos_list, orient='index').rename(columns={0:"x_pos"})

            y_pos_list = nx.get_node_attributes(G,'y_pos')
            df_y = pd.DataFrame.from_dict(y_pos_list, orient='index').rename(columns={0:"y_pos"})

            edges = [e for e in G.edges]

            df_node_tmp = pd.concat([df_x, df_y], axis=1)
            df_edge_tmp = pd.DataFrame(edges).rename(columns={0:"source", 1:"target"})

            x = torch.tensor(df_node_tmp.values, dtype=torch.float)
            edge_index = torch.tensor(np.array(df_edge_tmp[['source','target']]).T, dtype=torch.long)
            y = torch.tensor(np.array([int(labels[count])]))

            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)

        count += 1
        
    return dataset

def load_graph_loader(balanced):
    """
    Load training and test data for GCN into data loader 

    Args:
        balanced: which balanced data to use (false, oversampled or undersampled)
    Return:
        train_loader, test_loader, seminar_loader: data loaders used for training and testing the GCN model
    """
    train_loader, test_loader, seminar_loader = load_data_loader(balanced)

    if balanced == 'false':
        train_dataset = load_graph(train_loader, "Stroop", "train")
    else:
        train_dataset = load_graph(train_loader, "Stroop", "train", balanced==True)

    test_dataset = load_graph(test_loader, "Stroop", "test")
    seminar_dataset = load_graph(seminar_loader, "Seminar", "test")
    
    train_loader = torch_geometric.data.DataLoader(train_dataset, shuffle=True, batch_size=1)
    test_loader = torch_geometric.data.DataLoader(test_dataset, shuffle=False, batch_size=1)
    seminar_loader = torch_geometric.data.DataLoader(test_dataset, shuffle=False, batch_size=1)

    return train_loader, test_loader, seminar_loader

def load_data_loader(balanced):
    """
    Load training and test data for DL models into data loader 

    Args:
        balanced: which balanced data to use (false, oversampled or undersampled)
    Return:
        train_loader, test_loader, seminar_loader: data loaders used for training and testing the DL model
    """
    if balanced == 'false':
        train_dataset = MTDataset('Stroop', 'train')
    else:
        train_dataset = MTDataset('Stroop', 'train', balanced==True)
        
    test_dataset = MTDataset('Stroop', 'test')
    seminar_dataset = MTDataset('Seminar', 'test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    seminar_loader = DataLoader(seminar_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, seminar_loader
