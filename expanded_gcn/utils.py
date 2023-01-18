import numpy as np
import scipy.sparse as sp
import torch

'''similar to and derived from https://github.com/tkipf/pygcn/'''

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int8)
    return labels_onehot


def load_data(path, N=None, E=None):
    dataset = path[:-1]

    if dataset == 'nell':
        from torch_geometric.datasets import NELL
        nell = NELL(root=root+'nell/')
        nell = nell[0]
        features = nell.x
        features = features.to_scipy()
        features = features.tocsr()
        labels = nell.y.numpy()
        labels = encode_onehot(labels)
        idx = np.arange(labels.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = nell.edge_index
        edges_unordered = edges_unordered.T
        edges_unordered = edges_unordered.numpy()

    if dataset == 'random':
        features = np.random.randint(low=0, high=2, size=(N, 1))*1.0
        features = sp.csr_matrix(features)
        labels = np.random.randint(low=1, high=5, size=(N,))
        labels = encode_onehot(labels)
        idx = np.arange(labels.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}

        adj = (( sp.random(N, N, density = 1) > 1-(2*E/(N**2 - N)) )*1.0)
        adj.setdiag(0)
        
    else:
        idx_features_labels = np.genfromtxt("{}{}.content".format(root, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(root, dataset),
                                        dtype=np.int32)
        
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj_gcn = normalize(adj + sp.eye(adj.shape[0]))
    adj = normalize(adj)

    labels = np.where(labels)[1]

    features = torch.FloatTensor(np.array(features.todense()))

    adj_gcn = sparse_mx_to_torch_sparse_tensor(adj_gcn)
    labels = torch.LongTensor(labels)

    return adj_gcn, adj, features, labels


def get_split(labels_tensor, n_train, n_val):
    idx_train = []
    idx_val = []
    idx_test = []
    labels = labels_tensor.cpu().numpy()
    n_test = np.bincount(labels).min() - (n_train+n_val)
    for cls in np.unique(labels):
        shuffled_idx = np.random.permutation(np.where(labels==cls)[0])
        idx_train += shuffled_idx[:n_train].tolist()
        idx_val += shuffled_idx[n_train:(n_train+n_val)].tolist()
        idx_test += shuffled_idx[(n_train+n_val):n_test].tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)