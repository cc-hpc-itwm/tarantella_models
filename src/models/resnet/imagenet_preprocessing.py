# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
import logging
import os

try:
  from official.vision.image_classification.resnet import imagenet_preprocessing as tf_imagenet_preprocessing
except:
  from official.vision.image_classification import imagenet_preprocessing as tf_imagenet_preprocessing

import tensorflow as tf

NUM_CLASSES = 1001
NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

def input_fn(is_training,
             data_dir,
             batch_size,
             shuffle_seed = 42,
             datasets_num_private_threads=None,
             drop_remainder=False,
             training_dataset_cache=False,
             filenames=None,
             rank=0,
             comm_size=1,
             auto_distributed=True):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    shuffle_seed: seed to use for shuffling the data.
    datasets_num_private_threads: Number of private threads for tf.data.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    training_dataset_cache: Whether to cache the training dataset on workers.
      Typically used to improve training performance when training data is in
      remote storage and can fit into worker memory.
    filenames: Optional field for providing the file names of the TFRecords.

  Returns:
    A dataset that can be used for iteration.
  """
  if filenames is None:
    filenames = tf_imagenet_preprocessing.get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if not auto_distributed:
    logging.info('Dataset: apply sharding')
    # shard the dataset
    dataset.shard(num_shards=comm_size, index=rank)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=tf_imagenet_preprocessing._NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means that up to 10 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(tf.data.TFRecordDataset,
                               cycle_length=10,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training and training_dataset_cache:
    # Improve training performance when training data is in remote storage and
    # can fit into worker memory.
    dataset = dataset.cache()

  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (datasets_num_private_threads)
    dataset = dataset.with_options(options)
    logging.info('datasets_num_private_threads: %s', datasets_num_private_threads)

  # Parses the raw records into images and labels.
  dataset = dataset.map(lambda value: tf_imagenet_preprocessing.parse_record(value, is_training,
                                                                             dtype = tf.float32),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
