import argparse
import os
import datetime 

from models.resnet50 import imagenet_preprocessing
from models.resnet50 import resnet_model
from models.utils import keras_utils as utils

import train_loop
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

SHUFFLE_BUFFER = 6400 # nranks_per_node * 64 * imagesize * number batches < available memory
PREPROCESS_CYCLE_LENGTH = 16
PREPROCESS_NUM_THREADS = tf.data.experimental.AUTOTUNE
PREFETCH_NBATCHES = tf.data.AUTOTUNE  # should be ~ the number of devices

#tf.config.threading.set_inter_op_parallelism_threads(8)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", help="location of the ImageNet dataset")
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--train_epochs", type=int, default=90)
  parser.add_argument("--profile_dir", help="directory for profiles")
  parser.add_argument("--without-datapar",
                      action='store_true',
                      default = False)
  parser.add_argument("--shuffle-seed",
                      type = int,
                      default = 42)
  parser.add_argument("--val-freq",
                      type=int,
                      default = 1)
  parser.add_argument("--data-format",
                      help = "Reshape data into either 'channels_last' or 'channels_first' format",
                      default = "channels_last")
  parser.add_argument("--profile-runtimes", action='store_true',
                      default = False)
  parser.add_argument("--logging-freq", help="how often (in number of iterations) to record the runtimes per iteration",
                      type=int, default = 10)
  parser.add_argument("--print-freq", help="how often (in number of iterations) to print the recorded iteration runtimes",
                      type=int, default = 30)
  parser.add_argument("-s", "--strategy", type=str, default="data")
  parser.add_argument("--num_pipeline_stages", type=int, default=2)
  parser.add_argument("--num_partitions", type=int, default=2)

  args = parser.parse_args()
  if args.data_dir == None or not os.path.isdir(args.data_dir):
    sys.exit("ERROR: Cannot find images directory %s" % args.data_dir)
  return args  

args = parse_args()

if not args.without_datapar:
  import tarantella as tnt


def load_dataset(dataset_type,
                 data_dir,
                 batch_size,
                 dtype=tf.float32,
                 drop_remainder=False,
                 shuffle_seed = None):
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
  dataset = dataset.map(lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
                        num_parallel_calls=PREPROCESS_NUM_THREADS,
                        deterministic = False)
#  dataset = dataset.cache()
  dataset.prefetch(buffer_size=SHUFFLE_BUFFER)
  if is_training:
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER,
                              seed=shuffle_seed)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  
  dataset = dataset.prefetch(buffer_size=PREFETCH_NBATCHES)
  return dataset

def load_data(data_dir, batch_size, shuffle_seed = None):
  train_input_dataset = load_dataset(dataset_type='train', data_dir=data_dir,
                                     batch_size=batch_size, dtype=tf.float32,
                                     drop_remainder=True, shuffle_seed=shuffle_seed)
  val_input_dataset = load_dataset(dataset_type='validation', data_dir=data_dir,
                                   batch_size=batch_size, dtype=tf.float32,
                                   drop_remainder=True)
  return {"train" : train_input_dataset,
          "validation" : val_input_dataset }

def get_reference_compile_params():
  return {'optimizer' : tf.keras.optimizers.SGD(lr=train_loop.BASE_LEARNING_RATE, momentum=0.9),
          'loss' : tf.keras.losses.SparseCategoricalCrossentropy(),
          'metrics' : [tf.keras.metrics.SparseCategoricalAccuracy()]}

if __name__ == '__main__':

  have_datapar = False
  if not args.without_datapar:
    have_datapar = True

  rank = 0
  comm_size = 1
  if have_datapar:
    rank = tnt.get_rank()
    comm_size = tnt.get_size()


    strategy = tnt.ParallelStrategy.PIPELINING
    if args.strategy == "data":
      strategy = tnt.ParallelStrategy.DATA
    elif args.strategy == "all":
      strategy = tnt.ParallelStrategy.ALL


  callbacks = []
  callbacks.append(train_loop.LearningRateBatchScheduler(
                            train_loop.learning_rate_schedule,
                            batch_size = args.batch_size,
                            num_images = imagenet_preprocessing.NUM_IMAGES['train']))
  if args.profile_dir:
    # Start the training w/ logging
    log_dir = os.path.join(args.profile_dir, "gpi-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/rank" + str(rank))
    os.makedirs(log_dir, exist_ok = True)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    update_freq = 'batch',
                                                    profile_batch=(10,20),
                                                    histogram_freq=0))

  if rank == 0 and args.profile_runtimes:
    callbacks += [utils.RuntimeProfiler(batch_size = args.batch_size,
                                        logging_freq = args.logging_freq,
                                        print_freq = args.print_freq) ]

  model = resnet_model.resnet50(num_classes=imagenet_preprocessing.NUM_CLASSES,
                                num_partitions=args.num_partitions) # add split layers
  if have_datapar:
    model = tnt.Model(model,
                             parallel_strategy = strategy,
                             num_pipeline_stages = args.num_pipeline_stages)

  datasets = load_data(args.data_dir, batch_size = args.batch_size, shuffle_seed = args.shuffle_seed)
  model.compile(**get_reference_compile_params())

  history = model.fit(datasets['train'],
                      validation_data = datasets['validation'],
                      epochs=args.train_epochs,
                      callbacks=callbacks,
                      verbose=1)
  # eval_output = model.evaluate(datasets['validation'],
  #                             verbose=2)
  # if tnt.is_master_rank():
  #   print(f"Eval output = {eval_output}")
