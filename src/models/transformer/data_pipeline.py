# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Input pipeline for the transformer model to read, filter, and batch examples.

Two things to note in the pipeline:

1. Batching scheme

   The examples encoded in the TFRecord files contain data in the format:
     {"inputs": [variable length array of integers],
      "targets": [variable length array of integers]}
   Where integers in the arrays refer to tokens in the English and German vocab
   file (named `vocab.ende.32768`).

   Prior to batching, elements in the dataset are grouped by length (max between
   "inputs" and "targets" length). Each group is then batched such that:
     group_batch_size * length <= batch_size.

   Another way to view batch_size is the maximum number of tokens in each batch.

   Once batched, each element in the dataset will have the shape:
     {"inputs": [group_batch_size, padded_input_length],
      "targets": [group_batch_size, padded_target_length]}
   Lengths are padded to the longest "inputs" or "targets" sequence in the batch
   (padded_input_length and padded_target_length can be different).

   This batching scheme decreases the fraction of padding tokens per training
   batch, thus improving the training speed significantly.

2. Shuffling

   While training, the dataset is shuffled in two places in the code. The first
   is the list of training files. Second, while reading records using
   `parallel_interleave`, the `sloppy` argument is used to generate randomness
   in the order of the examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import tensorflow as tf

from official.nlp.transformer import data_pipeline
from official.utils.misc import model_helpers

# Buffer size for reading records from a TFRecord file. Each training file is
# 7.2 MB, so 8 MB allows an entire file to be kept in memory.
_READ_RECORD_BUFFER = 8 * 1000 * 1000

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _load_records(filename):
  """Read file and return a dataset of tf.Examples."""
  return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)

def _read_and_batch_from_files(file_pattern,
                               batch_size,
                               max_length,
                               max_io_parallelism,
                               shuffle,
                               shuffle_seed,
                               comm_size, rank):
  """Create dataset where each item is a dict of "inputs" and "targets".

  Args:
    file_pattern: String used to match the input TFRecord files.
    batch_size: Number of tokens per global batch of examples.
                The input is batched so that every batch has the shape
                [batch_size // max_length, max_length].
    max_length: Maximum number of tokens per example
    max_io_parallelism: Max number of cpu cores for parallel input processing.
    shuffle: If true, randomizes order of elements.
    shuffle_seed: Fixed seed to be used for shuffling.

  Returns:
    tf.data.Dataset object containing examples loaded from the files.
  """
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle, seed=shuffle_seed)

  # Read files and interleave results. When training, the order of the examples
  # will be non-deterministic.
  options = tf.data.Options()
  options.experimental_deterministic = False
  dataset = dataset.interleave(
      data_pipeline._load_records,
      cycle_length=max_io_parallelism,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

  # Parse each tf.Example into a dictionary
  # TODO: Look into prefetch_input_elements for performance optimization.
  dataset = dataset.map(
      data_pipeline._parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Remove examples where the input or target length exceeds the maximum length,
  dataset = dataset.filter(lambda x, y: data_pipeline._filter_max_length((x, y), max_length))
 
#   dataset = dataset.padded_batch(
#         # First calculate batch size (token number) per worker, then divide it
#         # into sentences, and finally expand to a global batch. It could prove
#         # the global batch divisble for distribution strategy.
#         int(batch_size // comm_size // max_length) * comm_size,
#         ([max_length], [max_length]),
#         drop_remainder=True)
#   dataset = dataset.unbatch()

#   dataset = dataset.shard(comm_size, rank)
#   dataset = dataset.padded_batch(
#         # First calculate batch size (token number) per worker, then divide it
#         # into sentences, and finally expand to a global batch. It could prove
#         # the global batch divisble for distribution strategy.
#         int(batch_size // comm_size // max_length),
#         ([max_length], [max_length]),
#         drop_remainder=True)

  dataset = dataset.padded_batch(
        # First calculate batch size (token number) per worker, then divide it
        # into sentences, and finally expand to a global batch. It could prove
        # the global batch divisble for distribution strategy.
        int(batch_size // max_length),
        ([max_length], [max_length]),
        drop_remainder=True)

  # Prefetch the next element to improve speed of input pipeline.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(data_pipeline.map_data_for_transformer_fn,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def train_input_fn(params, comm_size, rank):
  """Load and return dataset of batched examples for use during training."""
  file_pattern = os.path.join(params["data_dir"] or "", "*train*")
  return _read_and_batch_from_files(
      file_pattern,
      params["batch_size"],
      params["max_length"],
      params["max_io_parallelism"],
      shuffle=True,
      shuffle_seed = 42, comm_size = comm_size, rank=rank)


def eval_input_fn(params):
  """Load and return dataset of batched examples for use during evaluation."""
  file_pattern = os.path.join(params["data_dir"] or "", "*dev*")
  return _read_and_batch_from_files(
      file_pattern,
      params["batch_size"],
      params["max_length"],
      params["max_io_parallelism"],
      shuffle=False,
      shuffle_seed=42,comm_size = 1, rank = 0)


def map_data_for_transformer_fn(x, y):
  """Maps data for training, and handles weired behaviors for different vers."""
  # Will transform input x and targets y into tuple(x, y) as new model inputs.
  # For TF v2, the 2nd parameter is omitted to make Keras training work.
  return ((x, y),)