import torch
import torch.nn.functional as F
from sklearn import metrics
from munkres import Munkres
import numpy as np
import dgl
from collections import Counter

def g_from_torchsparse(adj):
    N = adj.shape[0]
    edges = adj.coalesce().indices() 
    src, dst = edges[0].tolist(), edges[1].tolist()
    weights = adj.coalesce().values()
    g = dgl.graph((src, dst), num_nodes=adj.shape[0])
    g.edata['weight'] = weights
    return g


def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    if is_sparse:
        m = edge_index.shape[1]
        weight = weight if weight is not None else torch.ones(m).to(edge_index.device)
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    else:
        adjacency = torch.zeros(N, N).to(edge_index.device)
        if weight is None:
            adjacency[edge_index[0], edge_index[1]] = 1
        else:
            adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    return adjacency

def adjacency2index(adjacency, weight=False):
    """_summary_

    Args:
        adjacency (torch.tensor): [N, N] matrix
    return:
        edge_index: [2, E]
        edge_weight: optional
    """
    adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight

    else:
        return edge_index
    
def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')
    
def decoding_from_assignment(assignmatrix):
    pred = assignmatrix.argmax(dim=1)
    return pred

class cluster_metrics:
    def __init__(self, trues, predicts):
        self.true_label = trues
        self.pred_label = predicts.cpu().numpy()

    def clusterAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)
        count_true = Counter(self.true_label)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        count_pred = Counter(self.pred_label)
        if numclass1 != numclass2:
            print(f"Class Not equal, class_true({numclass1})={count_true}, class_pred({numclass2})={count_pred}")
            return 0, 0, 0, 0, 0, 0, 0, self.pred_label

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        self.new_predicts = new_predict
        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, new_predict

    def evaluateFromLabel(self, use_acc=False):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        if use_acc:
            acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, new_predict = self.clusterAcc()
            return acc, nmi, f1_macro, ari, new_predict
        else:
            return 0, nmi, 0, ari, self.pred_label