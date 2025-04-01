from utility.parser import parse_args
from utility.dataset import Data
from utility.util import decoding_from_assignment, cluster_metrics
from model import DeSE
import torch
import torch.optim as optim
from time import time, strftime, localtime
import os
import numpy as np
import random
import dgl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx
from matplotlib.colors import ListedColormap
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.autograd.set_detect_anomaly(True)
print(torch.cuda.is_available())

def train(args):
    #prepare graph dataset and device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = 'cpu'
    print(device)
    dataset = Data(args.dataset, device)
    dataset.print_statistic()
    #dataset.print_degree()
    
    best_cluster_result = {}
    best_cluster = {'nmi': -0.001, 'ari': -0.001, 'acc': -0.001, 'f1': -0.001}
    #prepare model
    model = DeSE(args, dataset.feature, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    t0 = time()
    for epoch in range(args.epochs):
        t1 = time()
        s_dic, tree_node_embed_dic, g_dic = model(dataset.adj, dataset.feature, dataset.degrees)
        #se_loss = model.calculate_se_loss(s_dic, g_dic[args.height])
        t2 = time()
        se_loss = model.calculate_se_loss1()
        t3= time()
        lp_loss = model.calculate_lp_loss(g_dic[args.height], dataset.neg_edge_index, tree_node_embed_dic[args.height])
        t4 = time()
        loss = args.se_lamda * se_loss + args.lp_lamda * lp_loss
        optimizer.zero_grad() #梯度归零
        loss.backward()
        optimizer.step()

        if epoch % args.verbose == 0:
            pred = decoding_from_assignment(model.hard_dic[1])
            metrics = cluster_metrics(dataset.labels, pred)
            acc, nmi, f1, ari, new_pred = metrics.evaluateFromLabel(use_acc=True)
            if nmi > best_cluster['nmi']:
                best_cluster['nmi'] = nmi
                best_cluster_result['nmi'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_nmi.pt'.format(args.dataset, args.num_clusters_layer[0]))
            if ari > best_cluster['ari']:
                best_cluster['ari'] = ari
                best_cluster_result['ari'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_ari.pt'.format(args.dataset, args.num_clusters_layer[0]))
            if acc > best_cluster['acc']:
                best_cluster['acc'] = acc
                best_cluster_result['acc'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_acc.pt'.format(args.dataset, args.num_clusters_layer[0]))
            if f1 > best_cluster['f1']:
                best_cluster['f1'] = f1
                best_cluster_result['f1'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_f1.pt'.format(args.dataset, args.num_clusters_layer[0]))
            
            print(f"Epoch: {epoch} [{time()-t1:.3f}s], Loss: {loss.item():.6f} = {args.se_lamda} * {se_loss.item():.6f} + {args.lp_lamda} * {lp_loss.item():.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, ACC: {acc:.6f}, F1: {f1:.6f}")
            #print(f"train time: {t2-t1}; se_loss time: {t3-t2}; lp_loss time: {t4-t3}")
    #print('Total time: {:.3f}s'.format(time()-t0))
    print(f"Best NMI: {best_cluster_result['nmi']}, Best ARI: {best_cluster_result['ari']}, \nBest Cluster: {best_cluster}")
    print(args)

    save_path = './output/%s.result' % args.dataset
    f = open(save_path, 'a')
    f.write(f"lr={args.lr}, embed_dim={args.embed_dim}, se_lamda={args.se_lamda}, lp_lamda={args.lp_lamda}, k={args.k}, dropout={args.dropout}, beta_f={args.beta_f}, epochs={args.epochs}, height={args.height}, num_clusters={args.num_clusters_layer}, verbose={args.verbose}, activation={args.activation}, seed={args.seed} \n")
    f.write(f"--------Best NMI: {best_cluster_result['nmi']}, Best ARI: {best_cluster_result['ari']}, Best Cluster: {best_cluster} \n")
    f.close()
    return best_cluster


def draw_network(dataset):
    #prepare graph dataset and device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    dataset = Data(args.dataset, device)
    dataset.print_statistic()
    model = DeSE(args, dataset.feature, device).to(device)
    model.load_state_dict(torch.load('./save_model/{}_{}_acc.pt'.format(args.dataset, args.num_clusters_layer[0])))
    s_dic, tree_node_embed_dic, g_dic = model(dataset.adj, dataset.feature)
    pred = decoding_from_assignment(model.hard_dic[1])
    metrics = cluster_metrics(dataset.labels, pred)
    acc, nmi, f1, ari, new_pred = metrics.evaluateFromLabel(use_acc=True)
    print(new_pred)
    print('NMI: {:.6f}, ARI: {:.6f}, ACC: {:.6f}, F1: {:.6f}'.format(nmi, ari, acc, f1))
    '''
    nx_graph = dgl.to_networkx(g_dic[args.height])
    pos = nx.spring_layout(nx_graph, k=0.035, iterations=50, seed=5114)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.xlim(-1.05,1.05)
    plt.ylim(-1.05,1.05)
    color = []
    c =['darkred', 'royalblue', 'darkgreen', 'darkorange', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkviolet', 'darkslategray', 'darkturquoise', 'darkkhaki', 'darkolivegreen', 'darkorchid', 'darkseagreen', 'darkslateblue', 'darkgray', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet']
    for item in pred:
        color.append(c[item])
    print('drawing figure...')
    nx.draw_networkx_nodes(nx_graph, pos=pos, ax=ax, alpha=1, node_color=color, node_size=15)
    '''
    tsne = TSNE(n_components=2, random_state=5114)
    X_tsne = tsne.fit_transform(tree_node_embed_dic[args.height].detach().cpu().numpy())
    custom_cmap = ListedColormap(['#7d6181', '#d7aed3', '#46717c', '#a9d6d6', '#d5ba82', 
                                  '#aebfce', '#b36a6f', '#3c79b4', '#a8d9a2', '#ef7d31'])

    fig, ax = plt.subplots(figsize=(10,10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=new_pred, cmap=custom_cmap, s=50)
    plt.axis('off')
    plt.savefig("./figure/network_{}_{}.pdf".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')
    plt.savefig("./figure/network_{}_{}.png".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10,10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataset.labels, cmap=custom_cmap, s=50)
    plt.axis('off')
    plt.savefig("./figure/network_{}_{}_true.pdf".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')
    plt.savefig("./figure/network_{}_{}_true.png".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')


if __name__ == "__main__":
    args = parse_args()
    args.dataset = 'Photo'
    args.save = False
    if args.dataset == 'Cora':
        args.epochs = 600
        args.verbose = 20
        args.beta_f = 0.2 # 0.2, w/o feature beta_f=0
        args.dropout = 0.3  # 0.3
        args.embed_dim = 8 #8
        args.k = 1
        args.num_clusters_layer = [9]  # [9]
        args.lp_lamda = 5  #5
        args.se_lamda = 0.01  #0.01
        args.lr = 0.01
        args.seed = 406  #406
    elif args.dataset == 'Citeseer':
        args.epochs = 600
        args.verbose = 20
        args.beta_f = 0.1 # 0.1
        args.dropout = 0.05  #0.05
        args.embed_dim = 32  #32
        args.k = 1
        args.num_clusters_layer = [7] #[7]
        args.lp_lamda = 0.5  #0.5
        args.se_lamda = 0.01 #0.01
        args.lr = 0.001  #0.001
        args.seed = 262  #262
    elif args.dataset == 'Photo':
        args.epochs = 800  #800
        args.verbose = 20
        args.beta_f = 0.5  #0.4
        args.dropout = 0.3  #0.05
        args.embed_dim = 32  #64
        args.k = 1
        args.num_clusters_layer = [9]  #[9]
        args.lp_lamda = 0.5  #5
        args.se_lamda = 0.01  #0.01
        args.lr = 0.001
        args.seed = 110  #132
    elif args.dataset == 'Computers':   
        args.epochs = 800 #800
        args.verbose = 20
        args.beta_f = 0.3 #0.4
        args.dropout = 0.3
        args.embed_dim = 8  #32
        args.k = 1
        args.num_clusters_layer = [11]
        args.lp_lamda = 5  #0.5
        args.se_lamda = 0.5  #0.2
        args.lr = 0.005  #0.001
        args.seed = 398  #323
    elif args.dataset == 'Pubmed':
        args.epochs = 1000
        args.verbose = 20
        args.beta_f = 0.3
        args.dropout = 0.05
        args.embed_dim = 16
        args.k = 1
        args.num_clusters_layer = [5]
        args.lp_lamda = 1
        args.se_lamda = 0.5
        args.lr = 0.001
        args.seed = 335
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #使得网络相同输入下每次运行的输出固定
    dgl.seed(args.seed)
    train(args)
    #draw_network(args.dataset)
    