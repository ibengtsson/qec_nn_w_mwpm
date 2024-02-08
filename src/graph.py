"""Package with functions for creating graph representations of syndromes."""

import numpy as np
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Distance


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


def get_batch_of_graphs(
    syndromes,
    m_nearest_nodes,
    code_distance,
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

    x = torch.tensor(node_features[:, x_cols]).to(device)
    batch_labels = torch.tensor(node_features[:, batch_col]).long().to(device)
    pos = x[:, 2:]

    # get edge indices
    if m_nearest_nodes:
        edge_index = knn_graph(pos, m_nearest_nodes, batch=batch_labels)
    else:
        n = x.shape[0]
        edge_index = torch.cat(
            (
                torch.triu_indices(n, n, offset=1),
                torch.flipud(torch.triu_indices(n, n, offset=1)),
            ),
            axis=1,
        )

    # compute distances to get edge attributes
    width = code_distance + 1
    if experiment == "z":
        wrap_axis = 1
    else:
        wrap_axis = 0
    dist, eq_class = cylinder_distance(
        pos[edge_index[0], :], pos[edge_index[1], :], width, wrap_axis=wrap_axis
    )

    # cast eq_class to float32
    eq_class = eq_class.type(dtype=torch.float32)
    edge_attr = torch.stack((dist**power, eq_class), dim=1)

    return x, edge_index, edge_attr, batch_labels

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

        # map edges per graph to their original index in the edges array
        # detector_edges = [(detectors[tuple(x[edge[0], 2:].numpy())], detectors[tuple(x[edge[1], 2:].numpy())]) for edge in new_edges.T]

        edge_range = torch.arange(0, edges.shape[1])
        edge_indx.append(edge_range[edge_mask[0, :]])

    return (
        nodes_per_syndrome,
        edges_per_syndrome,
        weights_per_syndrome,
        classes_per_syndrome,
        edge_indx,
    )