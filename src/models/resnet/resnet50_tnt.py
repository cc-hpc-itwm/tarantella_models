from absl import app
from absl import flags
import logging
import os

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

import common
import imagenet_preprocessing

# Official Resnet50 model
try:
  from official.vision.image_classification.resnet import resnet_model
except:
  from official.vision.image_classification import resnet_model
from utils import RuntimeProfiler
# Enable Tarantella
import tarantella as tnt

def get_optimizer(batch_size):
  lr_schedule = common.PiecewiseConstantDecayWithWarmup(
                        batch_size=batch_size,
                        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
                        warmup_epochs=common.LR_SCHEDULE[0][1],
                        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
                        multipliers=list(p[0] for p in common.LR_SCHEDULE),
                        compute_lr_on_cpu=True)
  optimizer = gradient_descent_v2.SGD(learning_rate=lr_schedule, momentum=0.9)
  return optimizer

def main(_):
  flags_obj = flags.FLAGS

  # get rank and comm_size
  rank = tnt.get_rank()
  comm_size = tnt.get_size()

  # compute micro batch if the dataset is not automatically distributed by Tarantella
  if not flags_obj.auto_distributed:
    batch_size = flags_obj.batch_size // comm_size
  else:
    batch_size = flags_obj.batch_size

  # Load and preprocess datasets
  train_dataset = imagenet_preprocessing.input_fn(is_training=True,
                                                  data_dir=flags_obj.data_dir,
                                                  batch_size=batch_size,
                                                  shuffle_seed = 42,
                                                  drop_remainder=True,
                                                  rank=rank,
                                                  comm_size= comm_size,
                                                  auto_distributed=flags_obj.auto_distributed)
  validation_dataset = imagenet_preprocessing.input_fn(is_training=False,
                                                       data_dir=flags_obj.data_dir,
                                                       batch_size=batch_size,
                                                       shuffle_seed = 42,
                                                       drop_remainder=True,
                                                       rank=rank,
                                                       comm_size=comm_size,
                                                       auto_distributed=flags_obj.auto_distributed)

  # Create model and wrap it into a Tarantella model
  model = resnet_model.resnet50(num_classes=imagenet_preprocessing.NUM_CLASSES)
  model = tnt.Model(model)

  optimizer = get_optimizer(flags_obj.batch_size)
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=(['sparse_categorical_accuracy']))
  model.summary()

  callbacks = []
  if flags_obj.enable_tensorboard:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=flags_obj.model_dir,
                                                    profile_batch=2))
  if flags_obj.profile_runtime:
    callbacks.append(RuntimeProfiler(batch_size = batch_size,
                                    logging_freq = flags_obj.logging_freq,
                                    print_freq = flags_obj.print_freq))

  if flags_obj.enable_checkpoint_and_export:
    if flags_obj.model_dir is not None:
      ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
      callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))

  logging.info("Start training")
  kwargs = {'tnt_distribute_dataset': flags_obj.auto_distributed,
            'tnt_distribute_validation_dataset': flags_obj.auto_distributed}
  history = model.fit(train_dataset,
                      epochs=flags_obj.train_epochs,
                      callbacks=callbacks,
                      validation_data=validation_dataset,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=0,
                      **kwargs)
  logging.info("Train history: {}".format(history.history))

  kwargs = {'tnt_distribute_dataset': flags_obj.auto_distributed}
  eval_output = model.evaluate(validation_dataset,
                               verbose=0,
                               **kwargs)
  

if __name__ == '__main__':
  common.define_keras_flags()
  app.run(main)
