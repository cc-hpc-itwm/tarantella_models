import json
import os

import tensorflow as tf


def generate_nodelist(nodesfile):
  nodeslist = []
  with open(nodesfile, 'r') as f:
    for index, node in enumerate(f.readlines()):
      nodes_list += [f"{node}:222{index}"] # need to add a port number
  return nodeslist

def generate_tf_config_contents(nodesfile, rank):
  return json.dumps({
                    'cluster': { 'worker': generate_nodelist(nodesfile) },
                    'task': {'type': 'worker', 'index': rank} 
                    })

def tf_get_num_ranks(nodesfile):
  return len(generate_nodelist(nodesfile))

def tf_init_multiworker_strategy(nodesfile, rank):
  # Set the TF_CONFIG environment variable to configure the cluster setting. 
  os.environ['TF_CONFIG'] = generate_tf_config_contents(nodesfile, rank)
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  return strategy