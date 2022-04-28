import numpy as np
import time

from models.resnet50 import imagenet_preprocessing

import tarantella as tnt
import tensorflow as tf

SHUFFLE_BUFFER = 6400 # nranks_per_node * 64 * imagesize * number batches < available memory
PREPROCESS_CYCLE_LENGTH = 16
PREPROCESS_NUM_THREADS = tf.data.experimental.AUTOTUNE
PREFETCH_NBATCHES = tf.data.AUTOTUNE  # should be ~ the number of devices

data_dir="/home/DATA/ImageNet/"
dataset_type = "train"
shuffle_seed = 42
dtype = np.float32

alltime = time.time()
filenames = imagenet_preprocessing.get_filenames(dataset_type, data_dir)
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.cache()

is_training = (dataset_type in ["train"])
if is_training:
  # Shuffle the input files
  dataset = dataset.shuffle(buffer_size=imagenet_preprocessing._NUM_TRAIN_FILES,
                            seed = shuffle_seed)

# Convert to individual records.
dataset = dataset.interleave(tf.data.TFRecordDataset,
                              cycle_length=PREPROCESS_CYCLE_LENGTH,
                              num_parallel_calls=PREPROCESS_NUM_THREADS,
                              deterministic = False)
#dataset = dataset.take(195*batch_size) # for 10 input files
#dataset = dataset.cache()

# Parses the raw records into images and labels.
# dataset = dataset.map(lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
#                       num_parallel_calls=PREPROCESS_NUM_THREADS,
#                       deterministic = False)
dataset = dataset.batch(batch_size = 64, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=1)
  
tnt_dataset = tnt.data.Dataset(dataset = dataset,
                               num_ranks = 1,
                               rank = 0)



t = time.time()
tnt_dataset.distribute_dataset_across_ranks(apply_batch = True)
t = time.time() - t

samples_time = time.time()
n = tnt_dataset.num_samples
samples_time = time.time() - samples_time


dataset = dataset.batch(batch_size = 64, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=1)

tnt_dataset = tnt.data.Dataset(dataset = dataset,
                               num_ranks = 1,
                               rank = 0)

def func(i):
    i = i.numpy() # Decoding from the EagerTensor object
    x, y = lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype)(training_set[i])
    return x, y

z = list(range(n))
idataset = tf.data.Dataset.from_generator(lambda: z, tf.uint32)

idataset = idataset.shuffle(buffer_size=len(z), seed=0,
                          reshuffle_each_iteration=True)
new_dataset = idataset.map(lambda i: tf.py_function(func=func,
                                               inp=[i],
                                               Tout=[tf.uint32,
                                                     tf.float32]
                                               ),
                      num_parallel_calls=tf.data.AUTOTUNE)

dataset = dataset.batch(batch_size = 64, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=1)

tnt_dataset = tnt.data.Dataset(dataset = dataset,
                               num_ranks = 1,
                               rank = 0)

alltime = time.time() - alltime
print(f"num_samples={n} time={t} samples_time={samples_time} alltime={alltime}")