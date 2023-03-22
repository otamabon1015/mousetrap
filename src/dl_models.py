"""
@project: BA-Thesis: The best ways to catch a mouse without a mouse trap: Bridging the gap between behavioral research and machine learning methods
@author: Tamaki Ogura (ogura@cl.uni-heidelberg.de)
@filename: dl_models.py
@description: define all DL models used in this thesis
"""

import torch
import torch_geometric
from torch import flatten
from torch.autograd import Variable
from torch.nn import (LSTM, Conv1d, Dropout, BatchNorm1d,
                      Linear, MaxPool1d,  Module, ReLU,
                      TransformerEncoder, TransformerEncoderLayer)
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv


class CNN(Module):
    """
    CNN consisting of 
        CNN + ReLU + MaxPool (CNN filters: 16, kernel size: 2, stride: 1)
        FC + ReLU + Dropout (FC hidden units: 200, Dropout rate: 0.2)
        FC + ReLU + Dropout 
        FC
    """
    def __init__(self, num_channels=2, classes=2):
        super(CNN, self).__init__()
        self.conv = Conv1d(num_channels, 16, 2) 
        self.pool = MaxPool1d(2)
        self.fc1 = Linear(784, 200)
        self.fc2 = Linear(200, 200)
        self.fc3 = Linear(200, classes)
        self.dropout = Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))

        x = flatten(x, start_dim=0)
        x = x.unsqueeze(0)

        x = self.dropout(F.relu(self.fc1(x)))
        
        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)
        
        return x 

class DeepConvLSTM(Module):
    """
    DeepConvLSTM consisting of 
        CNN + ReLU + MaxPool (CNN filters: 16, kernel size: 2, stride: 1)
        LSTM + Dropout (LSTM hidden units: 10, Dropout rate: 0.2)
        FC
    """
    def __init__(self, num_channels=2, classes=2):
        super(DeepConvLSTM, self).__init__()
        self.conv = Conv1d(num_channels, 16, 2)
        self.pool = MaxPool1d(2)
        self.dropout = Dropout(p=0.2)
        self.fc = Linear(10, classes)
        self.lstm = LSTM(input_size=49, hidden_size=10,
                          num_layers=1, batch_first=True)
        
        self.h_0 = torch.randn(1, 1, 10) #hidden state
        self.c_0 = torch.randn(1, 1, 10) #internal state x.size(0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))

        x, (hn, cn) = self.lstm(x, (self.h_0, self.c_0))
        hn = hn.view(-1, 10)
        x = self.dropout(hn)

        x = self.fc(x)
        return x

class TransformerClassifier(Module):
    """
    Transfomer-encoder-based classifier consisting of 
        TransformerEncoder + MaxPool (TransformerEncoder heads: 5, dropout: 0.2)
        FC + ReLU + Dropout (FC hidden units: 64, Dropout rate: 0.2)
        FC + ReLU + Dropout 
        FC
    """
    
    def __init__(self, num_channels=100, classes=2):
        super(TransformerClassifier, self).__init__()
        
        self.encoder_layer = TransformerEncoderLayer(
        d_model=num_channels,   # the number of expected features in the input
        nhead=5,       # the number of heads in the multiheadattention models
        dim_feedforward=2048,
        dropout=0.2,
        batch_first=True
        )

        self.transformer_encoder = TransformerEncoder(
        self.encoder_layer.float() ,
        num_layers = 6,
        )
        
        self.linear1 = Linear(1, 64)
        self.linear2 = Linear(64, 64)
        self.linear3 = Linear(64, classes)

        self.pool = MaxPool1d(100)
        self.dropout = Dropout(p=0.2)
        
    def forward(self, x):
        x = self.pool(self.transformer_encoder(x))

        x = self.dropout(F.relu(self.linear1(x)))
        
        x = self.dropout(F.relu(self.linear2(x)))

        x = self.linear3(x)
        x = x[:, -1]
        
        return x

class GCN(Module):
    """
    GCN consisting of 
        GCN + ReLU + MaxPool (GCN hidden units: 64)
        GCN + ReLU + MaxPool 
        FC + ReLU + Dropout (FC hidden units: 64, Dropout rate: 0.2)
        FC + ReLU + Dropout 
        FC
    """
    def __init__(self, num_channels=2, classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_channels, 64)
        self.conv2 = GCNConv(64, 64)
        
        self.linear1 = Linear(64*100, 64)
        self.linear2 = Linear(64, 64)
        self.linear3 = Linear(64, classes)

        self.bn = BatchNorm1d(64)
        
    def forward(self, data):
        x = self.bn(F.relu(self.conv1(data.x, data.edge_index)))
        
        x = self.bn(F.relu(self.conv2(x, data.edge_index)))
        
        x = flatten(x, start_dim=0)
        x = x.unsqueeze(dim=0)
        
        x = F.relu(self.linear1(x))

        x = F.relu(self.linear2(x))
        
        x = self.linear3(x)
        
        return x
