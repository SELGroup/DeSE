import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from utility.util import select_activation, g_from_torchsparse
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import negative_sampling
from torch_scatter import scatter_sum
from time import time

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation=None):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = select_activation(activation)
    
    def forward(self, x):
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GCN_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, att=False):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.att_linear = nn.Linear(2*output_dim, 1)
        self.activation = select_activation(activation)
        self.att = att

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.att_linear(z2)
        return {'e': F.leaky_relu(a)}
    
    def message_func(self, edges):
        return {'h' : edges.src['h'], 'e' : edges.data['e']}
    
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h' : h}
    
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = self.linear(h)
            if self.att:
                graph.apply_edges(self.edge_attention)
                graph.update_all(self.message_func, self.reduce_func)
            else:
                graph.update_all(message_func = fn.copy_u('h', 'm'), reduce_func = fn.mean(msg='m',out='h'))
            h=graph.ndata.pop('h')
            if self.activation is not None:
                h=self.activation(h)
            return h

class Assign_layer(torch.nn.Module):
    def __init__(self, embed_dim, num_cluster, activation=None):
        super().__init__()
        self.GCN_emb=GCN_layer(embed_dim, embed_dim, activation, att=False)
        self.GCN_ass=GCN_layer(embed_dim, num_cluster, activation, att=True)
        self.num_cluster = num_cluster
    
    def update_graph_and_adj(self, adj, s):
        s_numpy = s.cpu().numpy()
        adj_s = s_numpy.T @ adj.tocsr() @ s_numpy
        adj_lil = sp.lil_matrix(adj_s)
        adj_lil.setdiag(0)
        adj_s = adj_lil.tocoo()
        graph = dgl.from_scipy(adj_s)
        graph = dgl.add_self_loop(graph)
        return graph, adj_s

    def forward(self, graph, adj, x):
        z = self.GCN_emb(graph, x)
        s = self.GCN_ass(graph, z)
        s = torch.softmax(s, dim=-1)
        x = s.t() @ z
        graph_higher, adj_higher = self.update_graph_and_adj(adj,s.detach())
        return s, x, graph_higher, adj_higher  #assignMatrix, father_x, new_graph, new_adj

def KNN(x, k):
    x1 = x.detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x1)
    distances, indices = nbrs.kneighbors(x1)
    rows = np.repeat(np.arange(x1.shape[0]), k)
    cols = indices.flatten()
    N =x1.shape[0]
    #adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    values = torch.ones(len(rows), dtype=torch.float)
    adj = torch.sparse_coo_tensor(indices=torch.tensor([rows, cols]), 
                                  values=values, 
                                  size=(N, N), 
                                  dtype=torch.float)
    adj = (adj + adj.t()) / 2.0
    return adj

class ASS(torch.nn.Module):
    def __init__(self, embed_dim, num_cluster, k, dropout=0.1, activation=None):
        super().__init__()
        self.GCN_emb=GCN_layer(embed_dim, embed_dim, activation, att=False)
        self.GCN_ass=GCN_layer(embed_dim, num_cluster, activation, att=True)
        self.num_cluster = num_cluster
        self.k = k
        self.mlp = MLP(embed_dim, embed_dim, embed_dim, dropout, activation)
    
    def forward(self, graph, x, adj_g):
        h = self.GCN_emb(graph, x)
        s = torch.softmax(self.GCN_ass(graph, x), dim=-1)
        e = s.t() @ h
        adj_g1 = s.t() @ adj_g @ s
        adj_f1 = KNN(self.mlp(e), self.k)
        return h, e, s, (adj_g1, adj_f1)


class DeSE(torch.nn.Module):
    def __init__(self, args, feature, device):
        super().__init__()
        self.num_node = feature.shape[0]
        self.input_dim = feature.shape[-1]
        self.height = args.height
        self.embed_dim = args.embed_dim
        self.activation = args.activation
        if args.num_clusters_layer is None:
            decay_rate = int(np.exp(np.log(self.num_nodes) / self.height)) if args.decay_rate is None else args.decay_rate
            num_clusters_layer = [int(self.num_nodes / (decay_rate ** i)) for i in range(1, self.height)]
        else:
            num_clusters_layer = args.num_clusters_layer
        self.mlp = MLP(self.input_dim, self.embed_dim, self.embed_dim)
        self.gnn = GCN_layer(self.input_dim, self.embed_dim, self.activation, att=False)
        self.assignlayers = nn.ModuleList([])
        for i in range(self.height - 1):
            self.assignlayers.append(ASS(self.embed_dim, num_clusters_layer[i], args.k, args.dropout, self.activation))
        self.device=device
        self.beta_f = args.beta_f
        self.k = args.k

    def hard(self, s_dic):
        assign_mat_dict = {self.height: torch.eye(self.num_node).to(self.device)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ s_dic[k + 1]
        self.hard_dic = {}
        for h, assign in assign_mat_dict.items():
            idx = assign.max(dim=1)[1]
            t = torch.zeros_like(assign)
            t[torch.arange(t.shape[0]), idx] = 1
            self.hard_dic[h] = t

    
    def forward(self, adj_g, feature):
        adj_f = KNN(self.mlp(feature), self.k)
        adj = adj_g + self.beta_f * adj_f
        g = g_from_torchsparse(adj)
        e = self.gnn(g, feature)
        s_dic = {} #layer2, layer1
        tree_node_embed_dic ={self.height: e.to(self.device)} #layer2, layer1
        g_dic ={self.height: g} #layer2, layer1
        for i, layer in enumerate(self.assignlayers):
            h, e, s, (adj_g, adj_f) = layer(g, e, adj_g)
            tree_node_embed_dic[self.height-i-1] = e.to(self.device)
            s_dic[self.height-i] = s.to(self.device)
            adj = adj_g + self.beta_f * adj_f
            if i==self.height-2:
                break
            g = g_from_torchsparse(adj.to_sparse())
            g_dic[self.height-i-1] = g.to(self.device)
        s_dic[1] = torch.ones(s.shape[-1], 1).to(self.device)
        self.hard(s_dic)
        self.s_dic = s_dic
        self.g_dic = g_dic
        return s_dic, tree_node_embed_dic, g_dic
    
    def calculate_se_loss(self, s_dic, g):
        #degrees = g.in_degrees()
        #t0 =time()
        edge_index = torch.stack(g.edges())
        weight = g.edata['weight']
        degrees = scatter_sum(weight, edge_index[0])
        vol_G = torch.sum(degrees).to(self.device)
        EPS = 1e-6
        assign_mat_dict = {self.height: torch.eye(self.num_node).to(self.device)} #each node at the bottom layer forms a cluster
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ s_dic[k + 1]  #assign_mat_dict[k] represent node assigned to which cluster at layer k: self.height->1
            vol_dict[k] = torch.einsum('ij, i->j', assign_mat_dict[k], degrees)  #vol_dict[k] represent vol of clusters at layer k: self.height->0
        se_loss = 0
        #t1 = time()
        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', s_dic[k], vol_dict[k - 1])  # (num_clusters_k, num_clusters_k-1) (num_clusters_k-1, ) -> (num_clusters_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (num_clusters_k, ) / (num_clusters_k, ) -> (num_clusters_k, )
            ass_i = assign_mat_dict[k][edge_index[0]]  # (E, num_clusters_k)
            ass_j = assign_mat_dict[k][edge_index[1]]  # Assignment of nodes at both ends of the edge to the cluster
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # ass_i * ass_j represent the propobalty that node_i node_j assigned to the same cluster: (E, num_clusters_k) (E, ) ->(num_clusters_k, ) total weight within the cluster
            delta_vol = vol_dict[k] - weight_sum    # (num_clusters_k, ) - (num_clusters_k, ) -> (num_clusters_k, )  total weight of cutting edges
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        #t2 = time()
        #print(t1-t0, t2-t1, t2-t0)
        return se_loss
    
    def calculate_se_loss1(self):
        g=self.g_dic[self.height]
        #t0 =time()
        edge_index = torch.stack(g.edges())
        weight = g.edata['weight']
        degrees = scatter_sum(weight, edge_index[0])
        vol_G = degrees.sum().to(self.device)
        EPS = 1e-6
        assign_mat_dict = {self.height: torch.eye(self.num_node, device=self.device)} #each node at the bottom layer forms a cluster
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ self.s_dic[k + 1]  #assign_mat_dict[k] represent node assigned to which cluster at layer k: self.height->1
            #vol_dict[k] = torch.einsum('ij, i->j', assign_mat_dict[k], degrees)  #vol_dict[k] represent vol of clusters at layer k: self.height->0
            vol_dict[k] = torch.matmul(assign_mat_dict[k].t(), degrees)
        se_loss = 0
        #t1 = time()
        for k in range(1, self.height + 1):
            #vol_parent = torch.einsum('ij, j->i', s_dic[k], vol_dict[k - 1])  # (num_clusters_k, num_clusters_k-1) (num_clusters_k-1, ) -> (num_clusters_k, )
            vol_parent = torch.matmul(self.s_dic[k], vol_dict[k - 1])
            log_vol_ratio_k = torch.log2_((vol_dict[k] + EPS) / (vol_parent + EPS))  # (num_clusters_k, ) / (num_clusters_k, ) -> (num_clusters_k, )
            ass_i = assign_mat_dict[k][edge_index[0]]  # (E, num_clusters_k)
            ass_j = assign_mat_dict[k][edge_index[1]]  # Assignment of nodes at both ends of the edge to the cluster
            weight_sum = torch.mv((ass_i * ass_j).t(), weight)  # ass_i * ass_j represent the propobalty that node_i node_j assigned to the same cluster: (E, num_clusters_k) (E, ) ->(num_clusters_k, ) total weight within the cluster
            delta_vol = vol_dict[k] - weight_sum    # (num_clusters_k, ) - (num_clusters_k, ) -> (num_clusters_k, )  total weight of cutting edges
            se_loss += torch.dot(delta_vol, log_vol_ratio_k)
        se_loss = -se_loss / vol_G
        #t2 = time()
        #print(t1-t0, t2-t1, t2-t0)
        return se_loss

    def calculate_dist(self, x, y):
        return torch.norm(x-y, p=2, dim=-1)

    def calculate_lp_loss(self, g, neg_edge_index, embedding):
        edge_index = torch.stack(g.edges())
        edge = torch.cat([edge_index, neg_edge_index], dim=-1)
        prob = self.calculate_dist(embedding[edge[0]], embedding[edge[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.cat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(self.device)
        lp_loss = F.binary_cross_entropy(prob, label)
        return lp_loss


class DSE1(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, height, num_nodes, num_clusters_layer, decay_rate, device, activation=None):
        super().__init__()
        if num_clusters_layer is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            num_clusters_layer = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        self.GCN_f1 = GCN_layer(input_dim, embed_dim, activation, att=False) #feature->embedding
        #self.GCN_f2 = GCN_layer(16, embed_dim, activation, att=False)
        self.assignlayers = nn.ModuleList([])
        for i in range(height - 1):
            self.assignlayers.append(Assign_layer(embed_dim, num_clusters_layer[i], activation))  #embedding->assignment
        self.height = height
        self.num_nodes = num_nodes
        self.device = device


    def hard(self, assignmatrix, tree_node_embed):
        self.embedding = {}
        for h, x in tree_node_embed.items():
            self.embedding[h] = x.detach()
        assign_mat_dict = {self.height: torch.eye(self.num_nodes).to(self.device)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ assignmatrix[k + 1]
        assignment = {}
        for h, assign in assign_mat_dict.items():
            idx = assign.max(dim=1)[1]
            t = torch.zeros_like(assign)
            t[torch.arange(t.shape[0]), idx] = 1
            assignment[h] = t
        return assignment

    def forward(self, graph, adj, feature):
        x=self.GCN_f1(graph, feature)  #feature->embedding
        #x=self.GCN_f2(graph, x)
        x=F.normalize(x, p=2, dim=-1) #normalize
        assignmatrix = {}  #store the assignment matrix of each layer: self.height->1
        tree_node_embed = {self.height: x.to(self.device)}  #store the embedding of each layer: self.height->0
        for i, layer in enumerate(self.assignlayers):  #embedding->assignment
            assign, x, graph, adj = layer(graph, adj, x)
            tree_node_embed[self.height-i-1] = x.to(self.device)
            assignmatrix[self.height-i] = assign.to(self.device)
        tree_node_embed[0] = torch.mean(x).to(self.device)
        assignmatrix[1] = torch.ones(assign.shape[-1], 1).to(self.device)
        self.hard(assignmatrix, tree_node_embed)
        return assignmatrix, tree_node_embed
    
    def calculate_se_loss(self, assignmatrix, degrees, edge_index, weight):
        vol_G = torch.sum(degrees).to(self.device)
        EPS = 1e-6
        assign_mat_dict = {self.height: torch.eye(self.num_nodes).to(self.device)} #each node at the bottom layer forms a cluster
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ assignmatrix[k + 1]  #assign_mat_dict[k] represent node assigned to which cluster at layer k: self.height->1
            vol_dict[k] = torch.einsum('ij, i->j', assign_mat_dict[k], degrees)  #vol_dict[k] represent vol of clusters at layer k: self.height->0
        se_loss = 0
        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', assignmatrix[k], vol_dict[k - 1])  # (num_clusters_k, num_clusters_k-1) (num_clusters_k-1, ) -> (num_clusters_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (num_clusters_k, ) / (num_clusters_k, ) -> (num_clusters_k, )
            ass_i = assign_mat_dict[k][edge_index[0]]  # (E, num_clusters_k)
            ass_j = assign_mat_dict[k][edge_index[1]]  # Assignment of nodes at both ends of the edge to the cluster
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # ass_i * ass_j represent the propobalty that node_i node_j assigned to the same cluster: (E, num_clusters_k) (E, ) ->(num_clusters_k, ) total weight within the cluster
            delta_vol = vol_dict[k] - weight_sum    # (num_clusters_k, ) - (num_clusters_k, ) -> (num_clusters_k, )  total weight of cutting edges
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss

    def calculate_entropy(self, p):
        entropy = -torch.sum(p * torch.log2(p))
        return entropy
    
    def calculate_onehot_loss(self, assignmatrix):
        onehot_loss = 0
        for k in range(2, self.height+1):
            entropy_values = [self.calculate_entropy(row) for row in assignmatrix[k]]
            if any(torch.isnan(e).item() for e in entropy_values):
                print(assignmatrix[k])
                raise ValueError("NaN found in entropy calculation")
            onehot_loss += torch.mean(torch.stack(entropy_values))
        return onehot_loss

    def calculate_regularizer_loss(self, tree_node_embed):
        regularizer = 0
        for k in range(1, self.height+1):
            embed = tree_node_embed[k]
            regularizer += embed.norm(2).pow(2)
        return regularizer

    def calculate_dist(self, x, y):
        return torch.norm(x-y, p=2, dim=-1)

    def calculate_lp_loss(self, edge_index, neg_edge_index, embedding):
        edge = torch.cat([edge_index, neg_edge_index], dim=-1)
        prob = self.calculate_dist(embedding[edge[0]], embedding[edge[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.cat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(self.device)
        lp_loss = F.binary_cross_entropy(prob, label)
        return lp_loss
