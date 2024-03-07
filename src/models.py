from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import sort_edge_index, softmax, one_hot
import numpy as np
import warnings

class SplitSyndromes(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, edges, edge_attr, detector_labels):

        node_range = torch.arange(0, detector_labels.shape[0]).to(edges.device)
        node_subset = node_range[detector_labels]

        valid_labels = torch.isin(edges, node_subset).sum(dim=0) == 2
        edges = edges[:, valid_labels]
        edge_attr = edge_attr[valid_labels, :]

        # now we want to remove the pairs (0-1, 1-0 etc)
        mask = edges[0, :] > edges[1, :]
        ind_range = torch.arange(edges.shape[1]).to(edges.device)
        edges, edge_attr = sort_edge_index(edges[:, ind_range[mask]], edge_attr[mask, :])
        
        return edges, edge_attr


class GraphNN(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64, 128],
        hidden_channels_MLP=[128, 64],
        n_node_features=5,
    ):
        super().__init__()

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Embedding layer for edge classes
        emb_dim = 64
        self.class_embd = nn.Linear(2, emb_dim)
        self.weight_embd = nn.Linear(1, emb_dim)
        # self.embedding = nn.Linear(2, emb_dim)
        
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 2 + 1
        channels = [transition_dim] + hidden_channels_MLP
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )
        
        # normalisation
        self.norms = nn.ModuleList(
            [
                nng.GraphNorm(in_channels) for in_channels in channels[1:]
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromes()
        
        # Activation function
        self.activation = torch.nn.Tanh()
        
        # Normalisation
        
        # self.norm = nng.GraphNorm(in_channels=hidden_channels_GCN[-1])

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
        warmup=False,
    ):

        w = edge_attr[:, [0]]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        w = edge_attr[:, [0]]
        # w = self.weight_embd(w)
        # w = self.activation(w)
        
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        c = self.class_embd(c)
        c = self.activation(c)
        
        # c = self.embedding(c)
        # c = self.activation(c)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        # edge_feat = torch.cat([x_src, w * c, x_dst], dim=-1)
        edge_feat = torch.cat([x_src, w, x_dst], dim=-1) 
        # edge_feat = torch.cat([x_src, edge_attr[:, [0]], x_dst], dim=-1)
        
        # send the edge features through linear layers
        _batch_labels = batch_labels[edges[0, :]]
        for layer, norm in zip(self.dense_layers, self.norms):
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
            edge_feat = norm(edge_feat, _batch_labels)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # if warmup, train network to do identity mapping
        if warmup:
            label = edge_attr[:, [0]]
            return edge_feat, label
        
        # otherwise, save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        # TEST
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        
        # min_inds = torch.argmin(edge_feat, dim=1)
        # edge_feat = edge_feat[range(n_edges // 2), min_inds]
        # edge_classes = edge_classes[range(n_edges // 2), min_inds]
        
        
        edges = edges[:, ::2]

        # normalise edge_weights per graph
        edge_batch = batch_labels[edges[0]]
        edge_feat = softmax(edge_feat, edge_batch)
        
        # # TEST
        # edge_classes = (edge_feat * (-1 * edge_classes)).mean(dim=1)
        # edge_feat = edge_feat.mean(dim=1)
        
        return edges, edge_feat, edge_classes

class GraphAttention(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64, 128],
        hidden_channels_MLP=[128, 64],
        n_node_features=5,
    ):
        super().__init__()
        
        # Embedding layer for edge classes
        emb_dim = 16
        self.embedding = nn.Linear(2, emb_dim)

        # Graph attention
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GATConv(in_channels, out_channels, edge_dim=emb_dim)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )
        
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 2
        channels = [transition_dim] + hidden_channels_MLP
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromes()
        
        # Activation function
        self.activation = torch.nn.Tanh()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # embed classes
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        c = self.embedding(c)
        c = self.activation(c)
        
        # graph attention
        for layer in self.graph_layers:
            x = layer(x, edges, c)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, x_dst], dim=-1)
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # otherwise, save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]

        # normalise edge_weights per graph
        edge_batch = batch_labels[edges[0]]
        edge_feat = softmax(edge_feat, edge_batch)
       
        return edges, edge_feat, edge_classes
