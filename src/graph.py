"""Package with functions for creating graph representations of syndromes."""

import numpy as np
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Distance
from torch_geometric.utils import unbatch


def get_node_list_3D(syndrome_3D):
    """
    Create two arrays, one containing the syndrome defects,
    and the other containing their corresponding contiguous
    indices in the matrix representation of the syndrome.
    """
    defect_indices_triple = np.nonzero(syndrome_3D)
    defects = syndrome_3D[defect_indices_triple]
    return defects, defect_indices_triple


def get_node_feature_matrix(defects, defect_indices_triple, num_node_features=None):
    """
    Creates a node feature matrix of dimensions
    (number_of_defects, number_of_node_features), where each row
    is the feature vector of a single node.
    The feature vector is defined as
    x = (X, Z, d_north, d_west, d_time)
        X: 1(0) if defect corresponds to a X(Z) stabilizer
        Z: 1(0) if defect corresponds to a Z(X) stabilizer
        d_north: distance to north boundary, i.e. row index in syndrome matrix
        d_west: distance to west boundary, i.e. column index in syndrome matrix
        d_time: distance in time from the first measurement
    """

    if num_node_features is None:
        num_node_features = 5  # By default, use 4 node features

    # Get defects (non_zero entries), defect indices (indices of defects in
    # flattened syndrome)
    # and defect_indices_tuple (indices in 3D syndrome) of the syndrome matrix

    num_defects = defects.shape[0]

    defect_indices_triple = np.transpose(np.array(defect_indices_triple))

    # get indices of x and z type defects, resp.
    x_defects = defects == 1
    z_defects = defects == 3

    # initialize node feature matrix
    node_features = np.zeros([num_defects, num_node_features])
    # defect is x type:
    node_features[x_defects, 0] = 1
    # distance of x tpe defect from northern and western boundary:
    node_features[x_defects, 2:] = defect_indices_triple[x_defects, :]

    # defect is z type:
    node_features[z_defects, 1] = 1
    # distance of z tpe defect from northern and western boundary:
    node_features[z_defects, 2:] = defect_indices_triple[z_defects, :]

    return node_features


# Function for creating a single graph as a PyG Data object
def get_3D_graph(syndrome_3D, target=None, m_nearest_nodes=None, power=None):
    """
    Form a graph from a repeated syndrome measurement where a node is added,
    each time the syndrome changes. The node features are 5D.
    """
    # get defect indices:
    defects, defect_indices_triple = get_node_list_3D(syndrome_3D)

    # Use helper function to create node feature matrix as torch.tensor
    # (X, Z, N-dist, W-dist, time-dist)
    X = get_node_feature_matrix(defects, defect_indices_triple, num_node_features=5)
    # set default power of inverted distances to 1
    if power is None:
        power = 2.0

    # construct the adjacency matrix!
    n_defects = len(defects)
    y_coord = defect_indices_triple[0].reshape(n_defects, 1)
    x_coord = defect_indices_triple[1].reshape(n_defects, 1)
    t_coord = defect_indices_triple[2].reshape(n_defects, 1)

    y_dist = np.abs(y_coord.T - y_coord)
    x_dist = np.abs(x_coord.T - x_coord)
    t_dist = np.abs(t_coord.T - t_coord)

    # inverse square of the supremum norm between two nodes
    Adj = np.maximum.reduce([y_dist, x_dist, t_dist])
    # set diagonal elements to nonzero to circumvent division by zero
    np.fill_diagonal(Adj, 1)
    # scale the edge weights
    Adj = 1.0 / Adj**power
    # set diagonal elements to zero to exclude self loops
    np.fill_diagonal(Adj, 0)

    # remove all but the m_nearest neighbours
    if m_nearest_nodes is not None:
        for ix, row in enumerate(Adj.T):
            # Do not remove edges if a node has (degree <= m)
            if np.count_nonzero(row) <= m_nearest_nodes:
                continue
            # Get indices of all nodes that are not the m nearest
            # Remove these edges by setting elements to 0 in adjacency matrix
            Adj.T[ix, np.argpartition(row, -m_nearest_nodes)[:-m_nearest_nodes]] = 0.0

    Adj = np.maximum(Adj, Adj.T)  # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(Adj)  # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(Adj)
    edge_attr = Adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)

    if target is not None:
        y = target.reshape(1, 1)
    else:
        y = None

    return [
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(
            edge_index.astype(
                np.int64,
            )
        ),
        torch.from_numpy(edge_attr.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    ]


def cylinder_distance(x, y, width, wrap_axis=1, manhattan=False):
    # x, y have coordinates (x, y, t)

    ds = torch.abs(x - y)
    eq_class = ds[:, wrap_axis] > 0.5 * width
    ds[eq_class, wrap_axis] = width - ds[eq_class, wrap_axis]

    if not manhattan:
        return torch.sqrt((ds**2).sum(axis=1)), eq_class
    else:
        return ds.sum(axis=1), eq_class
    
def inside_distance(x, y, manhattan=False):
    ds = x - y
    if not manhattan:
        return torch.sqrt((ds**2).sum(axis=1))
    else:
        return ds.sum(axis=1)
    
def outside_distance(x, y, width, wrap_axis=1, manhattan=False):
    ds = torch.abs(x - y)
    ds[:, wrap_axis] = width - ds[:, wrap_axis]
    if not manhattan:
        return torch.sqrt((ds**2).sum(axis=1))
    else:
        return ds.sum(axis=1)


def get_batch_of_graphs(
    syndromes,
    m_nearest_nodes,
    experiment="z",
    n_node_features=5,
    power=2.0,
    device=torch.device("cpu"),
):
    syndromes = syndromes.astype(np.float32)
    defect_inds = np.nonzero(syndromes)
    defects = syndromes[defect_inds]
    
    defect_inds = np.transpose(np.array(defect_inds))
    x_defects = defects == 1
    z_defects = defects == 3

    node_features = np.zeros((defects.shape[0], n_node_features + 1), dtype=np.float32)

    node_features[x_defects, 0] = 1
    node_features[x_defects, 2:] = defect_inds[x_defects, ...]
    node_features[z_defects, 1] = 1
    node_features[z_defects, 2:] = defect_inds[z_defects, ...]
    node_features.max(axis=0)
    x_cols = [0, 1, 3, 4, 5]
    batch_col = 2

    # x: (X-detector, Z-detector, N-dist, W-dist, time-dist)
    x = torch.tensor(node_features[:, x_cols]).to(device)
    batch_labels = torch.tensor(node_features[:, batch_col]).long().to(device)

    # get edge indices 
    if m_nearest_nodes:
        edge_index = knn_graph(x[:, 2:], m_nearest_nodes, batch=batch_labels)

    else:
        n = x.shape[0]
        edge_index = torch.cat(
            (
                torch.triu_indices(n, n, offset=1),
                torch.flipud(torch.triu_indices(n, n, offset=1)),
            ),
            axis=1,
        )

    # # compute distances to get edge attributes
    # width = code_distance + 1
    # if experiment == "z":
    #     wrap_axis = 0
    # else:
    #     wrap_axis = 1
    # dist, eq_class = cylinder_distance(
    #     pos[edge_index[0], :], pos[edge_index[1], :], width, wrap_axis=wrap_axis
    # )

    # # cast eq_class to float32
    # eq_class = eq_class.type(dtype=torch.float32)
    # edge_attr = torch.stack((dist**power, eq_class), dim=1)

    # create virtual nodes for the graphs with odd number of nodes (counted per Z/X-class)
    label = {"z": 3, "x": 1}
    even_odd = np.count_nonzero(syndromes == label[experiment], axis=(1, 2, 3)) & 1
    virtual_nodes = torch.zeros((np.sum(even_odd), n_node_features), dtype=torch.float32).to(device)
    label = {"z": 1, "x": 0}
    virtual_nodes[:, label[experiment]] = 1
    virtual_nodes[:, 2:] = -1 #FIX
    
    # # create batch labels
    virtual_batch_labels = (torch.arange(0, syndromes.shape[0])[even_odd.astype(bool)]).long().to(device)
    
    # add virtual nodes to node list and extend batch labels
    n_nodes_before = x.shape[0]
    x = torch.cat((x, virtual_nodes), axis=0)
    batch_labels = torch.cat((batch_labels, virtual_batch_labels), axis=0)
    n_nodes_after = x.shape[0]
    
    # do we need to sort? Will fuck up edge indices!
    # batch_labels, sort_indx = torch.sort(batch_labels)
    # x = x[sort_indx, :]
    
    # extend edge indices
    cum_node_sum = np.cumsum(np.count_nonzero(syndromes, axis=(1, 2, 3)))
    cum_node_sum = np.append(cum_node_sum, 0)
    
    ranges = [torch.arange(start, end) for start, end in zip(cum_node_sum[virtual_batch_labels - 1], cum_node_sum[virtual_batch_labels])]
    numbering = torch.arange(n_nodes_before, n_nodes_after)
    
    new_edges = torch.cat([torch.stack([target, torch.ones(target.shape, dtype=torch.int64) * num], dim=0) for target, num in zip(ranges, numbering)], dim=1).to(device)
    new_edges = torch.cat([new_edges, torch.flipud(new_edges)], dim=1)
    
    # append to existing indices
    edge_index = torch.cat([edge_index, new_edges], dim=1)
    
    # compute edge attributes (we'll have one edge for inner-distance and one for outer-distance)
    wrap_axis = {"x": 1, "z": 0}
    in_dist = inside_distance(x[edge_index[0, :], 2:], x[edge_index[1, :], 2:])
    out_dist = outside_distance(x[edge_index[0, :], 2:], x[edge_index[1, :], 2:], syndromes.shape[1], wrap_axis[experiment])
    dist = torch.cat([in_dist, out_dist], dim=0)
    
    # mark inner distance +1 and outer -1 
    in_mark = torch.ones_like(in_dist)
    out_mark = -1 * torch.ones_like(out_dist)
    mark = torch.cat([in_mark, out_mark], dim=0)
    
    # stack distance and marks together
    edge_attr = torch.stack([dist, mark], dim=1)
    
    # want to have two un-directed edges per node pair e.g. (1-0, 0-1, 1-0, 0-1), so let's double edge_index
    edge_index = torch.cat([edge_index, edge_index], dim=1)
    
    # mark which detectors that are of type experiment
    detector_labels = x[:, label[experiment]] == 1
    
    return x, edge_index, edge_attr, batch_labels, detector_labels

    # node_indices = torch.arange(0, x.shape[0])
    # nodes_per_graph = list(unbatch(node_indices,batch_labels))
    # connect_nodes = [nodes_per_graph[i] for i in virtual_batch_labels]
    # list_nodes = [x.tolist() for x in connect_nodes]
    # virtual_indices = torch.arange(x.shape[0],x.shape[0]+len(virtual_batch_labels))

    # num_new_edges = len(sum(list_nodes,[]))*2
    # virtual_edges = torch.zeros(2,num_new_edges,dtype=int)
    # count = 0
    # for i in range(0,len(virtual_indices)):
    #     curr_virtual = virtual_indices[i]
    #     curr_real = connect_nodes[i]
    #     for j in range(0,len(curr_real)):
    #         virtual_edges[0,count] = curr_virtual
    #         virtual_edges[1,count] = curr_real[j]
    #         virtual_edges[0,count+1] = curr_real[j]
    #         virtual_edges[1,count+1] = curr_virtual
    #         count += 2
                
    # #print(virtual_node_graph_nodes)
    # x = torch.cat((x, virtual_nodes), axis=0).to(device)
    # edge_index = torch.cat((edge_index, virtual_edges), axis=1).to(device)
    # virtual_attr = torch.ones((virtual_edges.shape[1],2),dtype=torch.float32)
    # edge_attr = torch.cat((edge_attr, virtual_attr), axis=0).to(device)
    # # return indicator labelling nodes belonging to experiment (X or Z-nodes)
    # detector_labels = x[:, label[experiment]] == 1
    
    # return x, edge_index, edge_attr, batch_labels, detector_labels

def extract_graphs(x, edges, edge_attr, batch_labels):

    node_range = torch.arange(0, x.shape[0])

    nodes_per_syndrome = []
    edges_per_syndrome = []
    weights_per_syndrome = []
    classes_per_syndrome = []
    edge_indx = []
    edge_weights = edge_attr[:, 0]
    edge_classes = edge_attr[:, 1]
    for i in range(batch_labels[-1] + 1):
        ind_range = torch.nonzero(batch_labels == i)

        # nodes
        nodes_per_syndrome.append(x[ind_range])

        # edges
        edge_mask = (edges >= node_range[ind_range[0]]) & (
            edges <= node_range[ind_range[-1]]
        )
        new_edges = edges[:, edge_mask[0, :]] - node_range[ind_range[0]]
        new_weights = edge_weights[edge_mask[0, :]]
        new_edge_classes = edge_classes[edge_mask[0, :]]

        edges_per_syndrome.append(new_edges)
        weights_per_syndrome.append(new_weights)
        classes_per_syndrome.append(new_edge_classes)

        edge_range = torch.arange(0, edges.shape[1])
        edge_indx.append(edge_range[edge_mask[0, :]])

    return (
        nodes_per_syndrome,
        edges_per_syndrome,
        weights_per_syndrome,
        classes_per_syndrome,
        edge_indx,
    )
    
def extract_edges(edges, edge_attr, batch_labels, node_range):

    edges_per_syndrome = []
    weights_per_syndrome = []
    classes_per_syndrome = []
    edge_indx = []
    edge_weights = edge_attr[:, 0]
    edge_classes = edge_attr[:, 1]
    for i in range(batch_labels[-1].item() + 1):
        ind_range = torch.nonzero(batch_labels == i)

        edge_mask = (edges >= node_range[ind_range[0]]) & (
            edges <= node_range[ind_range[-1]]
        )
        # new_edges = edges[:, edge_mask[0, :]] - node_range[ind_range[0]]
        new_edges = edges[:, edge_mask[0, :]]
        new_weights = edge_weights[edge_mask[0, :]]
        new_edge_classes = edge_classes[edge_mask[0, :]]

        edges_per_syndrome.append(new_edges)
        weights_per_syndrome.append(new_weights)
        classes_per_syndrome.append(new_edge_classes)

        edge_range = torch.arange(0, edges.shape[1])
        edge_indx.append(edge_range[edge_mask[0, :]])

    return (
        edges_per_syndrome,
        weights_per_syndrome,
        classes_per_syndrome,
        edge_indx,
    )



        


        
