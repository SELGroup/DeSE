import sys
sys.path.append('/workspace/DSE')
import torch
import numpy as np
from torch_scatter import scatter_sum
import torch_geometric.datasets
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import negative_sampling
from utility.util import index2adjacency
import dgl.data
import scipy.sparse as sp
import time
from collections import Counter

class Data:
    def __init__(self, dataset_name, device):
        self.name = dataset_name
        '''
        Cora, Citeseer, and Pubmed are citation network datasets, where nodes represent papers and edges represent citations.
        Computers and Photo are Amazon shopping datasets, where nodes represent items and edges represent similarity or relationships between items.
        CS and Physics are co-authorship network datasets, where nodes represent authors and edges represent co-authorship relationships.
        '''
        if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
            dataset = Planetoid(root='./datasets', name=dataset_name)
        elif dataset_name in ['Computers', 'Photo']:
            dataset = Amazon(root='./datasets', name=dataset_name)
        elif dataset_name in ["CS", "Physics"]:
            dataset = Coauthor(root='./datasets', name=dataset_name)
        data = dataset.data

        self.num_nodes = data.x.shape[0]
        self.feature = data.x.to(device)
        self.num_features = data.x.shape[1]
        self.num_edges = int(data.edge_index.shape[1]/2)
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = scatter_sum(self.weight, self.edge_index[0]).to(device)
        self.labels = data.y.tolist()
        self.num_classes = len(np.unique(self.labels))
        #self.adj = torch.coo_matrix((self.weight, (self.edge_index[0], self.edge_index[1])), shape=(self.num_nodes, self.num_nodes))
        self.adj = torch.sparse_coo_tensor(indices=self.edge_index, values=self.weight, size=(self.num_nodes, self.num_nodes))
        graph = dgl.graph((self.edge_index[0], self.edge_index[1]), num_nodes=self.num_nodes).to(device)
        self.weight = self.weight.to(device)
        self.edge_index = self.edge_index.to(device)
        self.graph = dgl.add_self_loop(graph)
        self.neg_edge_index = negative_sampling(self.edge_index)

    def print_statistic(self):
        print(f"Dataset Name: {self.name}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Number of features: {self.num_features}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Feature: {self.feature.shape}")
        print(f"edge_index: {self.edge_index.shape}")
        print(f"Graph: {self.graph}")


class Graph:
    def __init__(self, dataset_name):
        self.name = dataset_name
        '''
        Cora, Citeseer, and Pubmed are citation network datasets, where nodes represent papers and edges represent citations.
        Computers and Photo are Amazon shopping datasets, where nodes represent items and edges represent similarity or relationships between items.
        CS and Physics are co-authorship network datasets, where nodes represent authors and edges represent co-authorship relationships.
        '''
        if dataset_name == 'Cora':
            data = dgl.data.CoraGraphDataset()
        elif dataset_name == 'Citeseer':
            data = dgl.data.CiteseerGraphDataset()
        elif dataset_name == 'Pubmed':
            data = dgl.data.PubmedGraphDataset()
        elif dataset_name == 'Computers':
            data = dgl.data.AmazonCoBuyComputerDataset()
        elif dataset_name == 'Photo':
            data = dgl.data.AmazonCoBuyPhotoDataset()
        elif dataset_name == 'CS':
            data = dgl.data.CoauthorPhysicsDataset()
        elif dataset_name == 'Physics':
            data = dgl.data.CoauthorCSDataset()
        
        graph=data[0]

        self.num_nodes = graph.number_of_nodes()
        self.feature = graph.ndata['feat']
        self.num_features = graph.ndata['feat'].shape[1]
        self.num_edges = int(graph.number_of_edges()/2)
        self.edge_index = torch.stack(graph.edges(order='eid'))
        self.weight = torch.ones(self.edge_index.shape[1])
        self.degrees = graph.in_degrees()
        self.labels = graph.ndata['label'].tolist()
        self.num_classes = len(np.unique(self.labels))
        self.adj =sp.coo_matrix((self.weight, (self.edge_index[0], self.edge_index[1])), shape=(self.num_nodes, self.num_nodes))
        self.graph = dgl.add_self_loop(graph)


    def print_statistic(self):
        print(f"Dataset Name: {self.name}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Number of features: {self.num_features}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Feature: {self.feature.shape}")
        print(f"edge_index: {self.edge_index.shape}")
        print(f"Graph: {self.graph}")

        

    


