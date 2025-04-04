import random
from dataset import Data
import logging
from torch_scatter import scatter
import opt as opt
from opt import args
import torch
import numpy as np
from GAE import IGAE,IGAE_encoder
from time import time
from utils import setup_seed
from train import Train_gae
from sklearn.decomposition import PCA
from load_data import *


import warnings

from view_learner import ViewLearner

warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
setup_seed(np.random.randint(1000))


import pandas as pd

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

print("use cuda:{}".format(args.cuda))
device = torch.device("cuda:0" if opt.args.cuda else "cpu")

opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.model_save_path = 'model/model_save_gae/{}_gae.pkl'.format(opt.args.name)
opt.args.name = 'Cora'

t0 = time()
#print("Data: {}".format(opt.args.data_path))
#print("Label: {}".format(opt.args.label_path))
dataset = Data(opt.args.name, device)
dataset.print_statistic()
x = dataset.feature
y = np.array(dataset.labels)
adj = dataset.adj.to_dense()
edge_index1 = dataset.edge_index
#x = np.loadtxt(opt.args.data_path, dtype=float)
#y = np.loadtxt(opt.args.label_path, dtype=int)
#adj = torch.load('adj')
#adj = adj.to_dense()
#edge_index1=np.genfromtxt(opt.args.graph_save_path, dtype=np.int32)
#edge_index1 = edge_index1.transpose()


pca1 = PCA(n_components=opt.args.n_components)
x1 = pca1.fit_transform(x)
dataset = LoadDataset(x1)
data = torch.Tensor(dataset.x).to(device)



model_gae = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1,
        gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_enc_3=opt.args.gae_n_enc_3,
        n_input=data.shape[1]
    ).to(device)
#memory usage
total_params = sum(param.numel() for param in model_gae.parameters())
print(f"Total number of parameters: {total_params}")
all_float32 = all(param.dtype == torch.float32 for param in model_gae.parameters())
print(f"All parameters are float32: {all_float32}")

view_learner = ViewLearner(
        IGAE_encoder(gae_n_enc_1=opt.args.gae_n_enc_1,
                     gae_n_enc_2=opt.args.gae_n_enc_2,
                     gae_n_enc_3=opt.args.gae_n_enc_3,
                     n_input=data.shape[1]),
    ).to(device)
#memory usage
total_params += sum(param.numel() for param in view_learner.parameters())
print(f"Total number of parameters: {total_params}")
all_float32 = all(param.dtype == torch.float32 for param in view_learner.parameters())
print(f"All parameters are float32: {all_float32}")
memory_in_bytes = total_params * 4
memory_in_kb = memory_in_bytes / 1024
memory_in_mb = memory_in_kb / 1024
print(f"Memory Usage: {memory_in_bytes} Bytes")
print(f"Memory Usage: {memory_in_kb:.2f} KB")
print(f"Memory Usage: {memory_in_mb:.2f} MB")
exit()
Train_gae(model_gae,view_learner,data,adj.to(device), y, edge_index1)
print('Total time:', time() - t0)


