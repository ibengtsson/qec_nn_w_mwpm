from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import sort_edge_index, softmax, one_hot
import numpy as np
from qecsim.graphtools import mwpm
from src.graph import extract_edges

import warnings
warnings.filterwarnings('ignore', module='qecsim')


def mwpm_prediction(edges, weights, classes):

    classes = classes.astype(np.int32)
    # if only one edge, we only have one matching
    if edges.shape[1] == 1:
        flip = classes.sum() & 1
        return flip

    edges_w_weights = {tuple(sorted(x)): w for x, w in zip(edges.T, weights)}
    edges_w_classes = {tuple(sorted(x)): c for x, c in zip(edges.T, classes)}
    
    matched_edges = mwpm(edges_w_weights)

    # need to make sure matched_edges is sorted
    matched_edges = [tuple(sorted((x[0], x[1]))) for x in matched_edges]
    classes = np.array([edges_w_classes[edge] for edge in matched_edges])
    flip = classes.sum() & 1 

    return flip



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
        batch_labels
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
        self.activation = torch.nn.Tanh()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        batch_labels = 0
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
    
import torch.nn.functional as F
#from dgl.nn.pytorch import GATConv


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")
    
def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
    a = self.attn_fc(z2)
    return {"e": F.leaky_relu(a)}
    
def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox["e"], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    return {"h": h}


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h
    


class SimpleGraphNNV4(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[256, 512],
        hidden_channels_MLP=[256],
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