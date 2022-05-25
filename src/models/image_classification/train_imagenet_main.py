import argparse
import os
import datetime
import sys

from models.image_classification import imagenet_preprocessing
from models.image_classification import lr_scheduler
from models import utils

import tensorflow as tf

SHUFFLE_BUFFER = 6400 # nranks_per_node * 64 * imagesize * number batches < available memory
PREPROCESS_CYCLE_LENGTH = 8
PREPROCESS_NUM_THREADS = tf.data.experimental.AUTOTUNE
PREFETCH_NBATCHES = tf.data.AUTOTUNE  # should be ~ the number of devices

#tf.config.threading.set_inter_op_parallelism_threads(8)

cnn_models = {'resnet50': tf.keras.applications.resnet50.ResNet50,
              'resnet101': tf.keras.applications.resnet.ResNet101,
              'resnet152': tf.keras.applications.resnet.ResNet152,
              'efficientnetV2B0': tf.keras.applications.efficientnet_v2.EfficientNetV2B0,
              'efficientnetV2B1': tf.keras.applications.efficientnet_v2.EfficientNetV2B1,
              'efficientnetV2B2': tf.keras.applications.efficientnet_v2.EfficientNetV2B2,
              'efficientnetV2B3': tf.keras.applications.efficientnet_v2.EfficientNetV2B3,
              'efficientnetV2S': tf.keras.applications.efficientnet_v2.EfficientNetV2S,
              'efficientnetV2M': tf.keras.applications.efficientnet_v2.EfficientNetV2M,
              'efficientnetV2L': tf.keras.applications.efficientnet_v2.EfficientNetV2L,
              }

def add_bool_arg(parser, name, default=False):
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--' + name, dest=name, action='store_true')
  group.add_argument('--no-' + name, dest=name, action='store_false')
  parser.set_defaults(**{name:default})

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", help="location of the ImageNet dataset")
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--train_epochs", type=int, default=90)
  parser.add_argument("--train_num_samples", type=int, default=imagenet_preprocessing.NUM_IMAGES['train'])
  parser.add_argument("--val_num_samples", type=int, default=imagenet_preprocessing.NUM_IMAGES['validation'])
  parser.add_argument("--shuffle_seed", type = int, default = 42)
  parser.add_argument("--val_freq", type=int, default = 1)
  parser.add_argument("--verbose", type=int, default = 2)

  parser.add_argument("--profile_dir", help="directory for profiles")
  parser.add_argument("--logging_freq", help="how often (in number of iterations) to record the runtimes per iteration",
                      type=int, default = 10)
  parser.add_argument("--print_freq", help="how often (in number of iterations) to print the recorded iteration runtimes",
                      type=int, default = 30)
  add_bool_arg(parser, "profile_runtimes", default = False)

  add_bool_arg(parser, "distribute", default = True)
  parser.add_argument("--model_arch", type=str, default="resnet50",
                      help = f"Choose one of: {list(cnn_models.keys())}")
  parser.add_argument("--strategy", type=str, default="data")
  parser.add_argument("--num_pipeline_stages", type=int, default=1)
  parser.add_argument("--num_partitions", type=int, default=1)

  add_bool_arg(parser, "synthetic_data", default = False)
  add_bool_arg(parser, "drop_remainder", default = False)

  args = parser.parse_args()
  if not args.synthetic_data and (args.data_dir == None or not os.path.isdir(args.data_dir)):
   sys.exit(f"ERROR: Cannot find images directory {args.data_dir}")
  if not args.model_arch in list(cnn_models.keys()):
    sys.exit(f"ERROR: Model `{args.model_arch}` not supported (choose one of: {list(cnn_models.keys())})")
  return args  

args = parse_args()

if args.distribute:
  import tarantella as tnt

def load_synthetic_dataset(batch_size, num_samples, dtype, drop_remainder):
  images = tf.zeros(shape = (imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
                             imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
                             imagenet_preprocessing.NUM_CHANNELS),
                    dtype = dtype)
  labels = tf.zeros(shape = (), dtype = tf.int32)
  dataset = tf.data.Dataset.from_tensors((images, labels)).repeat(num_samples)
  dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)
  dataset = dataset.prefetch(buffer_size=PREFETCH_NBATCHES)
  return dataset

def load_dataset(dataset_type, data_dir, batch_size, num_samples,
                 dtype, drop_remainder, shuffle_seed):
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
  dataset = dataset.take(num_samples)
  # Parses the raw records into images and labels.
  dataset = dataset.map(lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
                        num_parallel_calls=PREPROCESS_NUM_THREADS,
                        deterministic = False)
  if is_training:
    dataset = dataset.shuffle(buffer_size=min(batch_size, 64*64),
                              seed=shuffle_seed)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(buffer_size=PREFETCH_NBATCHES)
  return dataset

def load_data(args):
  if args.synthetic_data:
    train_input_dataset = load_synthetic_dataset(num_samples = args.train_num_samples,
                                                 batch_size=args.batch_size,
                                                 dtype=tf.float32, drop_remainder=args.drop_remainder)
    val_input_dataset = load_synthetic_dataset(num_samples = args.val_num_samples,
                                               batch_size=args.batch_size,
                                               dtype=tf.float32, drop_remainder=args.drop_remainder)
  else:
    train_input_dataset = load_dataset(dataset_type='train',
                                       data_dir=args.data_dir, num_samples = args.train_num_samples,
                                       batch_size=args.batch_size, dtype=tf.float32,
                                       drop_remainder=args.drop_remainder,
                                       shuffle_seed=args.shuffle_seed)
    val_input_dataset = load_dataset(dataset_type='validation',
                                     data_dir=args.data_dir, num_samples = args.val_num_samples,
                                     batch_size=args.batch_size, dtype=tf.float32,
                                     drop_remainder=args.drop_remainder,
                                     shuffle_seed=args.shuffle_seed)
  return train_input_dataset, val_input_dataset

def get_reference_compile_params(num_ranks, num_samples, batch_size):
  num_steps_per_epoch = num_samples // batch_size
  lr_schedule = lr_scheduler.ExpDecayWithWarmupSchedule(
                            base_learning_rate=0.1,
                            num_ranks=num_ranks,
                            decay_steps=2.4 * num_steps_per_epoch,
                            decay_rate=0.97,
                            staircase=False,
                            warmup_steps=5 * num_steps_per_epoch,
                            warmup_rate=1.0)
  return {'optimizer' : tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                                momentum=0.9),
          'loss' : tf.keras.losses.SparseCategoricalCrossentropy(),
          'metrics' : [tf.keras.metrics.SparseCategoricalAccuracy()]}

if __name__ == '__main__':
  rank = 0
  num_ranks = 1
  if args.distribute:
    rank = tnt.get_rank()
    num_ranks = tnt.get_size()

    strategy = tnt.ParallelStrategy.PIPELINING
    if args.strategy == "data":
      strategy = tnt.ParallelStrategy.DATA
    elif args.strategy == "all":
      strategy = tnt.ParallelStrategy.ALL

  callbacks = []
  if args.profile_dir:
    # Start the training w/ logging
    log_dir = os.path.join(args.profile_dir, "tnt-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok = True)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    update_freq = 'batch',
                                                    profile_batch=(10,20),
                                                    histogram_freq=0))

  if args.profile_runtimes:
    profiler_callback = utils.RuntimeProfiler(batch_size = args.batch_size,
                                              logging_freq = args.logging_freq,
                                              print_freq = args.print_freq)
    if args.distribute:
      profiler_callback = tnt.keras.callbacks.Callback(profiler_callback,
                                                       run_on_all_ranks = False,
                                                       aggregate_logs = False)
    callbacks.append(profiler_callback)


  model_arch = cnn_models[args.model_arch]
  model = model_arch(include_top=True,
                     weights=None,
                     classes=1000,
                     input_shape=(224, 224, 3),
                     input_tensor=None,
                     pooling=None,
                     classifier_activation='softmax')

  if args.distribute:
    model = tnt.Model(model,
                      parallel_strategy = strategy,
                      num_pipeline_stages = args.num_pipeline_stages)

  train_dataset, val_dataset = load_data(args)
  model.compile(**get_reference_compile_params(num_ranks=num_ranks,
                                               num_samples = args.train_num_samples,
                                               batch_size=args.batch_size))

  history = model.fit(train_dataset,
                      validation_data = val_dataset,
                      validation_freq=args.val_freq,
                      epochs=args.train_epochs,
                      callbacks=callbacks,
                      verbose=args.verbose)

