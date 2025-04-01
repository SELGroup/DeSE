from dataset import Data
from model import EGAE, GAE
import torch
import warnings
import numpy as np
from time import time


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    name = 'Cora'
    #features, adjacency, labels = load_data(name)
    device = 'cpu'
    dataset = Data(name, device)
    dataset.print_statistic()
    features = dataset.feature
    adjacency = dataset.adj.to_dense()
    labels = np.array(dataset.labels)

    layers = [256, 128]
    acts = [torch.nn.functional.relu] * len(layers)
    # acts = [None, torch.nn.functional.relu]
    learning_rate = 10**-4*4
    pretrain_learning_rate = 0.001
    for coeff_reg in [0.001]:
        for alpha in [0.1]:
            t0 =time()
            print('========== alpha={}, reg={} =========='.format(alpha, coeff_reg))
            gae = EGAE(features, adjacency, labels, alpha, layers=layers, acts=acts,
                       max_epoch=600, max_iter=4, coeff_reg=coeff_reg, learning_rate=learning_rate)
            #memory usage
            total_params = sum(param.numel() for param in gae.parameters())
            print(f"Total number of parameters: {total_params}")
            all_float32 = all(param.dtype == torch.float32 for param in gae.parameters())
            print(f"All parameters are float32: {all_float32}")
            memory_in_bytes = total_params * 4
            memory_in_kb = memory_in_bytes / 1024
            memory_in_mb = memory_in_kb / 1024
            print(f"Memory Usage: {memory_in_bytes} Bytes")
            print(f"Memory Usage: {memory_in_kb:.2f} KB")
            print(f"Memory Usage: {memory_in_mb:.2f} MB")
            exit()
            gae.pretrain(10, learning_rate=pretrain_learning_rate)
            losses = gae.run()
            print('Total time:', time() - t0)
            #scio.savemat('losses_{}.mat'.format(name), {'losses': np.array(losses)})
