
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from dgl import batch
from model_architecture.TransfGNN import GNN


"""The MLP_revised class is used to build a feedforward neural network model with hidden layers and dropout.
It is the layer used to analyze ECFP strings and finally serves as the classification block."""
class MLP_revised(nn.Module):
    def __init__(self, n_feature, n_hidden: list, n_output=2, dropout=0.2):
        super(MLP_revised, self).__init__()

        assert isinstance(n_hidden, list), "n_hidden param must be a list"

        self.num_hidden_layers = len(n_hidden)
        self.layers = [n_feature] + n_hidden  # the layer list of NN except the output layer

        self.predict = nn.Sequential()
        self.predict.add_module('dropout_input', nn.Dropout(dropout))  # input_layers: dropout first
        for idx, (in_, out_) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.predict.add_module('linear{}'.format(idx), nn.Linear(in_, out_))  # add_module(dict): key -> nn.Module
            self.predict.add_module('relu{}'.format(idx), nn.ReLU())
            self.predict.add_module('dropout'.format(idx), nn.Dropout(dropout))

        self.predict.add_module('output', nn.Linear(self.layers[-1], n_output))

    def forward(self, x):
        return self.predict(x)

class myModel(nn.Module):
    def __init__(self, feature_size, model_params):
        super(myModel, self).__init__()

        self.feature_size = feature_size
        self.embedding_size = model_params["model_embedding_size"]
        self.dropout_rate = model_params["model_dropout_rate"]

        self.GATencoder = GNN(feature_size=self.feature_size, model_params=model_params)

        self.GATencoder_raw = GNN(feature_size=self.feature_size, model_params=model_params)
        
        # I use embedding_size*4 because I want to concatenate x_t and x_r, which according to my idea are the GNN representations before the Linear layer.
        # So x_t and x_r are of dimension embedding_size*2, and therefore their concatenation will be of dimension embedding_size*4.
        self.linear = nn.Linear(self.embedding_size * 4, self.embedding_size * 2)

        # I use a linear layer to bring the input of self.classify back to self.embedding_size*2.
        self.classify = MLP_revised(self.embedding_size * 2, [32],  # [64,32],[128]
                                    1, dropout=self.dropout_rate)  

        self.ln = nn.LayerNorm(self.embedding_size * 2)
        self.relu = nn.ReLU()
        self.mol_output = None

    def forward(self, data, device):
        # Processing raw features (mol_raws)    
        x_raw = data['mol_raws']['x']
        edge_index_raw = data['mol_raws']['edge_index']
        edge_attr_raw = data['mol_raws']['edge_attr']
        batch_index_raw = data['mol_raws']['batch']

        # Processing tree features (mol_trees)
        x_tree = data['mol_trees']['tree_node_features']
        edge_index_tree = data['mol_trees']['tree_edge_index']
        edge_attr_tree = data['mol_trees']['tree_edge_attr']
        batch_index_tree = data['mol_trees']['tree_batch_index']

        x_r, transformer_edge_index, attention_scores_mean = self.GATencoder_raw(x_raw, edge_attr_raw, edge_index_raw, batch_index_raw)

        # x_tree/edge_attr_tree/edge_index_tree/batch_index_tree sono gia' sul device corretto:
        # il chiamante (train_one_epoch/test) sposta l'intero tree_batch prima di costruire `data`.

        x_t, transformer_edge_index_tree, attention_scores_mean_tree = self.GATencoder(x_tree, edge_attr_tree, edge_index_tree, batch_index_tree)

        x = torch.cat([x_t, x_r], dim=-1)

        # The concatenation brings the dimension to embedding_size*4 so I pass it through a linear layer to reduce it to embedding_size*2
        x = self.linear(x) 
        if x.size(0) > 1:
            x = self.ln(x)
        
        y = self.classify(self.relu(x))
        return y, transformer_edge_index, attention_scores_mean, transformer_edge_index_tree, attention_scores_mean_tree
