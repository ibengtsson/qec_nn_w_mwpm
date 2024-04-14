from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import sort_edge_index, softmax, one_hot, unbatch, unbatch_edge_index
import numpy as np
import warnings
import math

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
    
class SplitSyndromesAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, edge_feat, edges, edge_classes, detector_labels):

        node_range = torch.arange(0, detector_labels.shape[0]).to(edges.device)
        node_subset = node_range[detector_labels]
              
        valid_labels = torch.isin(edges, node_subset).sum(dim=-1) != 2
        edges[valid_labels, :] = 0
        edge_feat[valid_labels, :] = 0
        edge_classes[valid_labels, :] = 0
        
        return edge_feat, edges, edge_classes




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
    
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 2 + 1 + 2
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

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        
        w = edge_attr[:, [0]]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, w, c, x_dst], dim=-1) 
        
        # send the edge features through linear layers
        _batch_labels = batch_labels[edges[0, :]]
        for layer, norm in zip(self.dense_layers, self.norms):
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
            edge_feat = norm(edge_feat, _batch_labels)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
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

class GatConvNN(nn.Module):

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

class GraphNNV2(nn.Module):

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
        
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 3 + 3
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
        
        # Activation functions
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        w = edge_attr[:, [0]]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        x_pool = nng.global_mean_pool(x, batch_labels)
        inds = batch_labels[edges[0, :]]
        x_emb = torch.cat([x_pool[inds], w, c], dim=-1)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, x_dst, x_emb], dim=-1) 
        edge_feat_p_batch = list(unbatch(edge_feat, inds))
        edge_feat = torch.nested.as_nested_tensor(edge_feat_p_batch)
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # seperate data
        edges_p_batch = list(unbatch(edges, inds, dim=1))
        edges = torch.nested.as_nested_tensor(edges_p_batch)
        
        edge_classes_p_batch = list(unbatch(edge_attr[:, [1]], inds))
        edge_classes = torch.nested.as_nested_tensor(edge_classes_p_batch)
        
        return edges.unbind(), edge_feat.unbind(), edge_classes.unbind()
    
class SimpleGraphNN(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        hidden_channels_MLP=[64],
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
    
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 2 + 1 + 2
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
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
    ):

        
        w = edge_attr[:, [0]]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, w, c, x_dst], dim=-1) 
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]
        
        return edges, edge_feat, edge_classes
    
class SimpleGraphNNV2(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        hidden_channels_MLP=[64],
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
        
        # Embedding layer
        self.embed = nn.Linear(3, hidden_channels_GCN[-1])
    
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 3
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
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
    ):

        
        w = edge_attr[:, [0]]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        emb = self.embed(torch.cat([w, c], dim=-1))
        emb = self.activation(emb)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, emb, x_dst], dim=-1) 
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]
        
        return edges, edge_feat, edge_classes
    
class SimpleGraphNNV3(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        hidden_channels_MLP=[64],
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
    
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 3 + 3
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
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        
        w = edge_attr[:, [0]]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        # graph level information
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        x_pool = nng.global_mean_pool(x, batch_labels)
        inds = batch_labels[edges[0, :]]
        x_emb = torch.cat([x_pool[inds], w, c], dim=-1)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, x_dst, x_emb], dim=-1) 
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]
        
        return edges, edge_feat, edge_classes
    
class SimpleGraphNNV4(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        hidden_channels_MLP=[64],
        n_node_features=5,
    ):
        super().__init__()
        
        # Weight embedding
        self.weight_emb = nn.Linear(3, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )
    
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 3 + 3
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
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        w = self.weight_emb(torch.cat([w, c], dim=-1))
        w = self.activation(w)
        
        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        # graph level information
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        x_pool = nng.global_mean_pool(x, batch_labels)
        inds = batch_labels[edges[0, :]]
        x_emb = torch.cat([x_pool[inds], w, c], dim=-1)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, x_dst, x_emb], dim=-1) 
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]
        
        return edges, edge_feat, edge_classes
    
class SimpleGraphNNV5(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        hidden_channels_MLP=[64],
        n_node_features=5,
    ):
        super().__init__()

        # Weight embedding
        self.weight_emb_one = nn.Linear(7, 64)
        self.weight_emb_two = nn.Linear(64, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Edge embedding
        self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])

        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 4
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
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        src = x[edges[0, :], :2]
        dst = x[edges[1, :], :2]
        emb = torch.cat([src, w, c, dst], dim=-1)

        w = self.weight_emb_one(emb)
        w = self.activation(w)
        w = self.weight_emb_two(w)
        w = self.activation(w)

        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)

        # create an embedding for weights and classes on an edge level
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        emb = self.edge_emb(torch.cat([w, c], dim=-1))
        emb = self.activation(emb)

        # aggregate graph level information to aid edge weight generation
        x_pool = nng.global_mean_pool(x, batch_labels)
        inds = batch_labels[edges[0, :]]
        x_pool = x_pool[inds, :]

        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, x_dst, x_pool, emb], dim=-1) 

        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)

        # output
        edge_feat = self.output_layer(edge_feat)

        # save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)

        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]

        return edges, edge_feat, edge_classes
    
class SimpleGraphNNV6(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        hidden_channels_MLP=[64],
        n_node_features=5,
    ):
        super().__init__()
        
        # Weight embedding
        self.weight_emb = nn.Linear(7, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )
        
        # Edge embedding
        self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])
    
        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 4
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
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        src = x[edges[0, :], :2]
        dst = x[edges[1, :], :2]
        emb = torch.cat([src, w, c, dst], dim=-1)
        
        w = self.weight_emb(emb)
        w = self.activation(w)
        
        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        
        # create an embedding for weights and classes on an edge level
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        emb = self.edge_emb(torch.cat([w, c], dim=-1))
        emb = self.activation(emb)
        
        # aggregate graph level information to aid edge weight generation
        x_pool = nng.global_mean_pool(x, batch_labels)
        inds = batch_labels[edges[0, :]]
        x_pool = x_pool[inds, :]
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, x_dst, x_pool, emb], dim=-1) 
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=1)
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]
        
        return edges, edge_feat, edge_classes

class SelfAttention(nn.Module):
    
    def __init__(
        self,
        embd_dim,
        num_heads=1,
    ):  
        
        super().__init__()
        assert embd_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # linear transformation for Q, V, K (stack together for efficiency)
        self.qkv_proj = nn.Linear(embd_dim, embd_dim * 3)
        self.mh = nn.MultiheadAttention(embd_dim, num_heads=num_heads, batch_first=True)
        
    # assume x: (batch_size, sequence_length, feature_dim)
    def forward(self, x, mask=None):

        # linear transformations
        qkv = self.qkv_proj(x)
        
        # extract q, k and v
        q, k, v = qkv.chunk(3, dim=-1)

        # run attention
        attn_out, _ = self.mh(q, k, v, key_padding_mask=mask, need_weights=False)

        return attn_out
    
class GraphAttention(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        n_node_features=5,
    ):
        super().__init__()

        # Weight embedding
        self.weight_emb_one = nn.Linear(7, 64)
        self.weight_emb_two = nn.Linear(64, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Edge embedding
        self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])

        # Attention layer
        attention_dim = hidden_channels_GCN[-1] * 3
        self.attention = SelfAttention(attention_dim)

        # Output layer
        self.output_layer = nn.Linear(attention_dim, 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromes()

        # Activation function
        self.activation = torch.nn.ReLU()
        
        # Normalisations
        self.graph_norm = nng.GraphNorm(channels[-1])

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        src = x[edges[0, :], :2]
        dst = x[edges[1, :], :2]
        emb = torch.cat([src, w, c, dst], dim=-1)

        w = self.weight_emb_one(emb)
        w = self.activation(w)
        w = self.weight_emb_two(w)
        w = self.activation(w)

        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # normalisation
        x = self.graph_norm(x, batch_labels)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)

        # create an embedding for weights and classes on an edge level
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        emb = self.edge_emb(torch.cat([w, c], dim=-1))
        emb = self.activation(emb)

        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, emb, x_dst], dim=-1) 
        
        # unbatch data and pad sequences
        inds = batch_labels[edges[0, :]]
        edge_feat = unbatch(edge_feat, inds)
        edge_feat = torch.nn.utils.rnn.pad_sequence(edge_feat, batch_first=True)

        src_edges = unbatch(edges[0, :], inds)
        trg_edges = unbatch(edges[1, :], inds)

        src_edges = torch.nn.utils.rnn.pad_sequence(src_edges, batch_first=True)
        trg_edges = torch.nn.utils.rnn.pad_sequence(trg_edges, batch_first=True)
        edges = torch.stack([src_edges, trg_edges], dim=-1)

        edge_classes = unbatch(edge_attr[:, [1]], inds)
        edge_classes = torch.nn.utils.rnn.pad_sequence(edge_classes, batch_first=True)
        
        # create mask ensuring attention is not applied for the padding
        mask = (edges[:, :, 0] == edges[:, :, 1])

        # send the edge features through an attention layer
        edge_feat = self.attention(edge_feat, mask=mask)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        edge_feat = torch.cat([edge_feat[:, ::2, :], edge_feat[:, 1::2, :]], dim=2)
        edge_classes = torch.cat([edge_classes[:, ::2, :], edge_classes[:, 1::2, :]], dim=2)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=2, keepdim=True)
        edge_classes = torch.gather(edge_classes, -1, min_inds)
        
        edges = edges[:, ::2, :]

        return edges, edge_feat, edge_classes
    
class GraphAttentionV2(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        n_node_features=5,
    ):
        super().__init__()
        
        # Weight embedding
        self.weight_emb = nn.Linear(3, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Edge embedding
        self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])

        # Attention layer
        attention_dim = hidden_channels_GCN[-1] * 3
        self.attention = SelfAttention(attention_dim)

        # Output layer
        self.output_layer = nn.Linear(attention_dim, 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromesAttention()

        # Activation function
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):
        
        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        w = self.weight_emb(torch.cat([w, c], dim=-1))
        w = self.activation(w)

        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
            
        # remove duplicates 0-1, 1-0 and sort (needed for later unbatch)
        mask = edges[0, :] > edges[1, :]
        ind_range = torch.arange(edges.shape[1]).to(edges.device)
        edges = edges[:, ind_range[mask]]
        edge_attr = edge_attr[ind_range[mask], :]
        edges, edge_attr = sort_edge_index(edges, edge_attr)
        
        
        # make an edge feature vector 
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        w = self.edge_emb(torch.cat([edge_attr[:, [0]], c], dim=-1))
        w = self.activation(w)
        
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, w, x_dst], dim=-1) 
        
        # unbatch data and pad sequences
        inds = batch_labels[edges[0, :]]
        edge_feat = unbatch(edge_feat, inds)
        edge_feat = torch.nn.utils.rnn.pad_sequence(edge_feat, batch_first=True)

        src_edges = unbatch(edges[0, :], inds)
        trg_edges = unbatch(edges[1, :], inds)

        src_edges = torch.nn.utils.rnn.pad_sequence(src_edges, batch_first=True)
        trg_edges = torch.nn.utils.rnn.pad_sequence(trg_edges, batch_first=True)
        edges = torch.stack([src_edges, trg_edges], dim=-1)

        edge_classes = edge_attr[:, [1]]
        edge_classes = unbatch(edge_classes, inds)
        edge_classes = torch.nn.utils.rnn.pad_sequence(edge_classes, batch_first=True)
        
        # create mask ensuring attention is not applied for the padding
        mask = (edges[:, :, 0] == edges[:, :, 1])

        # send the edge features through an attention layer and then split the syndrome
        edge_feat = self.attention(edge_feat, mask=mask)
        edge_feat, edges, edge_classes = self.split_syndromes(edge_feat, edges, edge_classes, detector_labels)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        edge_feat = torch.cat([edge_feat[:, ::2, :], edge_feat[:, 1::2, :]], dim=2)
        edge_classes = torch.cat([edge_classes[:, ::2, :], edge_classes[:, 1::2, :]], dim=2)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=2, keepdim=True)
        edge_classes = torch.gather(edge_classes, -1, min_inds)

        edges = torch.stack([edges[:, ::2, :], edges[:, 1::2, :]], dim=-1)
        src_edges = torch.gather(edges[:, :, 0], -1, min_inds)
        trg_edges = torch.gather(edges[:, :, 1], -1, min_inds)
        edges = torch.cat([src_edges, trg_edges], axis=-1)

        return edges, edge_feat, edge_classes
    
class GraphAttentionV3(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        n_node_features=5,
        num_heads=1,
    ):
        super().__init__()

        # Weight embedding
        self.weight_emb = nn.Linear(3, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Edge embedding
        self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])

        # Attention layer
        attention_dim = hidden_channels_GCN[-1] * 3
        self.attention = SelfAttention(attention_dim, num_heads=num_heads)

        # Output layer
        self.output_layer = nn.Linear(attention_dim, 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromes()

        # Activation function
        self.activation = torch.nn.ReLU()
        
        # Normalisations
        self.graph_norm = nng.GraphNorm(channels[-1])
        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        w = self.weight_emb(torch.cat([w, c], dim=-1))
        w = self.activation(w)

        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # normalisation
        x = self.graph_norm(x, batch_labels)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)

        # create an embedding for weights and classes on an edge level
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        emb = self.edge_emb(torch.cat([w, c], dim=-1))
        emb = self.activation(emb)

        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, emb, x_dst], dim=-1) 
        
        # unbatch data and pad sequences
        inds = batch_labels[edges[0, :]]
        edge_feat = unbatch(edge_feat, inds)
        edge_feat = torch.nn.utils.rnn.pad_sequence(edge_feat, batch_first=True)

        src_edges = unbatch(edges[0, :], inds)
        trg_edges = unbatch(edges[1, :], inds)

        src_edges = torch.nn.utils.rnn.pad_sequence(src_edges, batch_first=True)
        trg_edges = torch.nn.utils.rnn.pad_sequence(trg_edges, batch_first=True)
        edges = torch.stack([src_edges, trg_edges], dim=-1)

        edge_classes = unbatch(edge_attr[:, [1]], inds)
        edge_classes = torch.nn.utils.rnn.pad_sequence(edge_classes, batch_first=True)
        
        # create mask ensuring attention is not applied for the padding
        mask = (edges[:, :, 0] == edges[:, :, 1])

        # send the edge features through an attention layer
        attention_out = self.attention(edge_feat, mask=mask)
        
        # add/norm with skip connection
        edge_feat = edge_feat + attention_out
        edge_feat = self.layer_norm(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        edge_feat = torch.cat([edge_feat[:, ::2, :], edge_feat[:, 1::2, :]], dim=2)
        edge_classes = torch.cat([edge_classes[:, ::2, :], edge_classes[:, 1::2, :]], dim=2)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=2, keepdim=True)
        edge_classes = torch.gather(edge_classes, -1, min_inds)
        
        edges = edges[:, ::2, :]

        return edges, edge_feat, edge_classes
    
class GraphAttentionV4(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64],
        n_node_features=5,
    ):
        super().__init__()

        # Weight embedding
        self.weight_emb = nn.Linear(3, 1)

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Edge embedding
        self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])

        # Attention layer
        attention_dim = hidden_channels_GCN[-1] * 3
        self.attention_one = SelfAttention(attention_dim)
        self.attention_two = SelfAttention(attention_dim)

        # Output layer
        self.output_layer = nn.Linear(attention_dim, 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromes()

        # Activation function
        self.activation = torch.nn.ReLU()
        
        # Normalisations
        self.graph_norm = nng.GraphNorm(channels[-1])
        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels,
    ):

        # weight embedding
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        
        w = self.weight_emb(torch.cat([w, c], dim=-1))
        w = self.activation(w)

        # graph layers
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)

        # normalisation
        x = self.graph_norm(x, batch_labels)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)

        # create an embedding for weights and classes on an edge level
        w = edge_attr[:, [0]]
        c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
        emb = self.edge_emb(torch.cat([w, c], dim=-1))
        emb = self.activation(emb)

        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, emb, x_dst], dim=-1) 
        
        # unbatch data and pad sequences
        inds = batch_labels[edges[0, :]]
        edge_feat = unbatch(edge_feat, inds)
        edge_feat = torch.nn.utils.rnn.pad_sequence(edge_feat, batch_first=True)

        src_edges = unbatch(edges[0, :], inds)
        trg_edges = unbatch(edges[1, :], inds)

        src_edges = torch.nn.utils.rnn.pad_sequence(src_edges, batch_first=True)
        trg_edges = torch.nn.utils.rnn.pad_sequence(trg_edges, batch_first=True)
        edges = torch.stack([src_edges, trg_edges], dim=-1)

        edge_classes = unbatch(edge_attr[:, [1]], inds)
        edge_classes = torch.nn.utils.rnn.pad_sequence(edge_classes, batch_first=True)
        
        # create mask ensuring attention is not applied for the padding
        mask = (edges[:, :, 0] == edges[:, :, 1])

        # send the edge features through an attention layer
        attention_out = self.attention_one(edge_feat, mask=mask)

        # add/norm with skip connection
        edge_feat = edge_feat + attention_out
        edge_feat = self.layer_norm(edge_feat)
        
        # do another attention layer
        attention_out = self.attention_two(edge_feat, mask=mask)
        
        # add/norm with skip connection
        edge_feat = edge_feat + attention_out
        edge_feat = self.layer_norm(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # save the edges with minimum weights (for each duplicate edge)
        edge_feat = torch.cat([edge_feat[:, ::2, :], edge_feat[:, 1::2, :]], dim=2)
        edge_classes = torch.cat([edge_classes[:, ::2, :], edge_classes[:, 1::2, :]], dim=2)
        
        edge_feat, min_inds = torch.min(edge_feat, dim=2, keepdim=True)
        edge_classes = torch.gather(edge_classes, -1, min_inds)
        
        edges = edges[:, ::2, :]

        return edges, edge_feat, edge_classes

        
# class GraphAttention(nn.Module):

#     def __init__(
#         self,
#         hidden_channels_GCN=[32, 64],
#         n_node_features=5,
#     ):
#         super().__init__()

#         # Weight embedding
#         self.weight_emb_one = nn.Linear(7, 64)
#         self.weight_emb_two = nn.Linear(64, 1)

#         # GCN layers
#         channels = [n_node_features] + hidden_channels_GCN
#         self.graph_layers = nn.ModuleList(
#             [
#                 nng.GraphConv(in_channels, out_channels)
#                 for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
#             ]
#         )

#         # Edge embedding
#         self.edge_emb = nn.Linear(3, hidden_channels_GCN[-1])

#         # Attention layer
#         attention_dim = hidden_channels_GCN[-1] * 4
#         self.attention = SelfAttention(attention_dim)

#         # Output layer
#         self.output_layer = nn.Linear(attention_dim, 1)

#         # Layer to split syndrome into X (Z)-graphs
#         self.split_syndromes = SplitSyndromes()

#         # Activation function
#         self.activation = torch.nn.ReLU()

#     def forward(
#         self,
#         x,
#         edges,
#         edge_attr,
#         detector_labels,
#         batch_labels,
#     ):

#         # weight embedding
#         w = edge_attr[:, [0]]
#         c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
#         src = x[edges[0, :], :2]
#         dst = x[edges[1, :], :2]
#         emb = torch.cat([src, w, c, dst], dim=-1)

#         w = self.weight_emb_one(emb)
#         w = self.activation(w)
#         w = self.weight_emb_two(w)
#         w = self.activation(w)

#         # graph layers
#         for layer in self.graph_layers:
#             x = layer(x, edges, w)
#             x = self.activation(x)

#         # split syndromes so only X (Z) nodes remain and create an edge embedding
#         edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)

#         # create an embedding for weights and classes on an edge level
#         w = edge_attr[:, [0]]
#         c = one_hot((edge_attr[:, 1]).to(dtype=torch.long), num_classes=2)
#         emb = self.edge_emb(torch.cat([w, c], dim=-1))
#         emb = self.activation(emb)

#         # aggregate graph level information to aid edge weight generation
#         x_pool = nng.global_mean_pool(x, batch_labels)
#         inds = batch_labels[edges[0, :]]
#         x_pool = x_pool[inds, :]

#         x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
#         edge_feat = torch.cat([x_src, x_dst, x_pool, emb], dim=-1) 
        
#         # unbatch data and pad sequences
#         edge_feat = unbatch(edge_feat, inds)
#         edge_feat = torch.nn.utils.rnn.pad_sequence(edge_feat, batch_first=True)

#         src_edges = unbatch(edges[0, :], inds)
#         trg_edges = unbatch(edges[1, :], inds)

#         src_edges = torch.nn.utils.rnn.pad_sequence(src_edges, batch_first=True)
#         trg_edges = torch.nn.utils.rnn.pad_sequence(trg_edges, batch_first=True)
#         edges = torch.stack([src_edges, trg_edges], dim=-1)

#         edge_classes = unbatch(edge_attr[:, [1]], inds)
#         edge_classes = torch.nn.utils.rnn.pad_sequence(edge_classes, batch_first=True)

#         # send the edge features through an attention layer
#         edge_feat = self.attention(edge_feat)
        
#         # output
#         edge_feat = self.output_layer(edge_feat)
        
#         # save the edges with minimum weights (for each duplicate edge)
#         edge_feat = torch.cat([edge_feat[:, ::2, :], edge_feat[:, 1::2, :]], dim=2)
#         edge_classes = torch.cat([edge_classes[:, ::2, :], edge_classes[:, 1::2, :]], dim=2)
        
#         edge_feat, min_inds = torch.min(edge_feat, dim=2, keepdim=True)
#         edge_classes = torch.gather(edge_classes, -1, min_inds)
        
#         edges = edges[:, ::2, :]

#         return edges, edge_feat, edge_classes

# class MultiheadAttention(nn.Module):
    
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         num_heads=1,
#     ):  
        
#         super().__init__()
#         assert output_dim % num_heads == 0, "Output dimension must be divisible by number of heads"
        
#         self.output_dim = output_dim
#         self.num_heads = num_heads
#         self.head_dim = output_dim // num_heads
    
#         # linear transformation for Q, V, K (stack together for efficiency)
#         self.output_dim = output_dim
#         self.qkv_proj = nn.Linear(input_dim, output_dim * 3)
#         self.mh = nn.MultiheadAttention(output_dim, num_heads=num_heads, batch_first=True)
#         self.out_proj = nn.Linear(output_dim, output_dim)
    
#     # assume x: (batch_size, sequence_length, feature_dim)
#     def forward(self, x, mask=None):
#         batch_size, seq_length, _ = x.shape
        
#         # linear transformations
#         qkv = self.qkv_proj(x)
        
#         # extract q, k and v
        
        
        
#         qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
#         qkv = qkv.permute(0, 2, 1, 3)
#         queries, keys, values = qkv.chunk(3, dim=-1)
        
#         # scaled self-attention
#         d_k = queries.shape[-1]
#         attn_logits = torch.matmul(queries, keys.transpose(-2, -1))
#         attn_logits = attn_logits / math.sqrt(d_k)
        
#         print(x.shape)
#         print(queries.shape, keys.shape)
#         print(attn_logits.shape)
#         if mask is not None:
#             print(mask.view(batch_size, 1, 1, seq_length).shape)
#             print(mask.view(batch_size, 1, 1, seq_length)[0, 0, 0, :])
#             attn_logits = attn_logits.masked_fill(mask.view(batch_size, 1, 1, seq_length), -9e15)
#         attention = torch.nn.functional.softmax(attn_logits, dim=-1)
        
#         values = torch.matmul(attention, values)
#         values = values.permute(0, 2, 1, 3)
#         values = values.reshape(batch_size, seq_length, self.output_dim)
#         output = self.out_proj(values)
        
#         return output
