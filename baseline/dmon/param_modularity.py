# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TODO(tsitsulin): add headers, tests, and improve style."""
import os
from time import time
from absl import app
from absl import flags
from sklearn.metrics import normalized_mutual_info_score
import tensorflow.compat.v2 as tf
from utilities.util import cluster_metrics
from models.gcn_modularity import gcn_modularity
from utilities.graph import load_kipf_data
from utilities.graph import load_npz_to_sparse_graph
from utilities.graph import normalize_graph
from utilities.graph import scipy_to_tf
from utilities.metrics import conductance
from utilities.metrics import modularity
from utilities.metrics import precision
from utilities.metrics import recall
from utilities.metrics import accuracy_score
from utilities.dataset import Data
import torch
import scipy.sparse

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'Photo', 'Dataset name')
flags.DEFINE_string('graph_path', None, 'Input graph path')
flags.DEFINE_string('output_path', None, 'Output results path')
flags.DEFINE_string('architecture', '[128_256_128]', 'Network architecture')
flags.DEFINE_string('load_strategy', 'schur', 'Graph format')
flags.DEFINE_string('postfix', '', 'File postfix')
flags.DEFINE_float(
    'orthogonality_regularization',
    1,
    'Orthogonality regularization parameter',
    lower_bound=0)
flags.DEFINE_float(
    'cluster_size_regularization',
    1,
    'Cluster size regularization parameter',
    lower_bound=0)
flags.DEFINE_float(
    'dropout_rate',
    0,
    'Orthogonality regularization parameter',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_integer('n_clusters', 10, 'Number of clusters', lower_bound=0)
flags.DEFINE_integer('n_epochs', 600, 'Number of epochs', lower_bound=0)
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate', lower_bound=0)


def format_filename():
  graph_name = os.path.split(FLAGS.graph_path)[1]
  architecture_str = FLAGS.architecture.strip('[]')
  return (f'{FLAGS.output_path}/{graph_name}-'
          f'nclusters-{FLAGS.n_clusters}-'
          f'architecture-{architecture_str}-'
          f'ortho-{FLAGS.orthogonality_regularization}-'
          f'clustersize-{FLAGS.cluster_size_regularization}-'
          f'dropout-{FLAGS.dropout_rate}-'
          f'lr-{FLAGS.learning_rate}-'
          f'epochs-{FLAGS.n_epochs}'
          f'postfix-{FLAGS.postfix}'
          '.txt')

def main(argv):
  t0 = time()
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  device = 'cpu'
  dataset = Data(FLAGS.dataset, device)
  dataset.print_statistic()

  n_nodes = dataset.num_nodes
  feature_size = dataset.num_features
  #feature_size = feature_size.item() if isinstance(feature_size, torch.Tensor) else feature_size
  architecture = [int(x) for x in FLAGS.architecture.strip('[]').split('_')]
  architecture.append(FLAGS.n_clusters)

  sparse_tensor = dataset.adj
  sparse_tensor = sparse_tensor.coalesce()
  rows, cols = sparse_tensor.indices().numpy()
  data = sparse_tensor.values().numpy()
  shape = sparse_tensor.size()
  # 转换为 scipy.sparse.coo_matrix
  adjacency = scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape)
  

  features = dataset.feature
  sparse_tensor = features.to_sparse()
  indices = sparse_tensor.indices().t().numpy()  # 转置为 (nnz, 2) 格式
  values = sparse_tensor.values().numpy()        # 提取非零值
  dense_shape = sparse_tensor.size()             # 获取稠密张量的形状
  # 3. 构造 TensorFlow 的 SparseTensor
  features= tf.SparseTensor(
      indices=indices,  # 非零元素的索引
      values=values,    # 非零元素的值
      dense_shape=dense_shape  # 稠密张量的形状
  )

  labels = dataset.labels
  graph_clean = scipy_to_tf(adjacency)
  graph_clean_normalized = scipy_to_tf(
      normalize_graph(adjacency.copy(), normalized=True))

  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = gcn_modularity(
      [input_features, input_graph, input_adjacency],
      architecture,
      dropout_rate=FLAGS.dropout_rate,
      orthogonality_regularization=FLAGS.orthogonality_regularization,
      cluster_size_regularization=FLAGS.cluster_size_regularization)
  #memory usage
  total_params = model.count_params()
  print(f"Total number of parameters: {total_params}")
  memory_in_bytes = total_params * 4
  memory_in_kb = memory_in_bytes / 1024
  memory_in_mb = memory_in_kb / 1024
  print(f"Memory Usage: {memory_in_bytes} Bytes")
  print(f"Memory Usage: {memory_in_kb:.2f} KB")
  print(f"Memory Usage: {memory_in_mb:.2f} MB")
  #exit()

  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  model.compile(optimizer, None)

  for epoch in range(FLAGS.n_epochs):
    #print(type(features), type(graph_clean_normalized), type(graph_clean))
    loss_values, grads = grad(model,
                              [features, graph_clean_normalized, graph_clean])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
          ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))

  _, assignments = model([features, graph_clean_normalized, graph_clean],
                         training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)
  adjacency=adjacency.toarray()
  print('Conductance:', conductance(adjacency, clusters))
  print('Modularity:', modularity(adjacency, clusters))
  metrics = cluster_metrics(labels, clusters)
  acc, nmi, f1, ari, new_pred = metrics.evaluateFromLabel(use_acc=True)
  print('NMI:', nmi)
  print('ARI:', ari)
  print('Accuracy:', acc)
  print('F1:', f1)
  print('Total time:', time() - t0)
  


if __name__ == '__main__':
  app.run(main)
  
