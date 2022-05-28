import json
import os

import tensorflow as tf

def generate_nodelist(nodesfile, nnp):
  nodeslist = []
  with open(nodesfile, 'r') as f:
    nodes = f.readlines()

  unique_nodes = list(dict.fromkeys(nodes))
  nested_nodes = [[node]*nnp for node in unique_nodes ]
  nodes = [n for nodes in nested_nodes for n in nodes]
  for index, node in enumerate(nodes):
    nodeslist += [f"{node.strip()}:32{100+index}"] # need to add a port number
  return nodeslist

def generate_tf_config_contents(nodesfile, rank, nnp):
  json_config = json.dumps({
                            'cluster': { 'worker': generate_nodelist(nodesfile, nnp) },
                            'task': {'type': 'worker', 'index': rank} 
                           })
  return json_config

def tf_get_num_ranks(nodesfile, nnp):
  return len(generate_nodelist(nodesfile, nnp))

def tf_init_multiworker_strategy(nodesfile, rank, nnp):
  # Set the TF_CONFIG environment variable to configure the cluster setting. 
  os.environ['TF_CONFIG'] = generate_tf_config_contents(nodesfile, rank, nnp)
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  return strategy
