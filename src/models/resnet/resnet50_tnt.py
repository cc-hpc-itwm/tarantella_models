import logging
import os

from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification.resnet import common as tf_common
from official.vision.image_classification.resnet import imagenet_preprocessing as tf_imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_model

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

import common
import imagenet_preprocessing

import tarantella as tnt
tnt.init()

def get_optimizer(batch_size):
  lr_schedule = tf_common.PiecewiseConstantDecayWithWarmup(
                        batch_size=batch_size,
                        epoch_size=tf_imagenet_preprocessing.NUM_IMAGES['train'],
                        warmup_epochs=common.LR_SCHEDULE[0][1],
                        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
                        multipliers=list(p[0] for p in common.LR_SCHEDULE),
                        compute_lr_on_cpu=True)
  optimizer = gradient_descent_v2.SGD(learning_rate=lr_schedule, momentum=0.9)
  return optimizer

def main(_):
  flags_obj = flags.FLAGS

  # Create model
  model = resnet_model.resnet50(num_classes=tf_imagenet_preprocessing.NUM_CLASSES)
  model = tnt.Model(model)

  optimizer = get_optimizer(flags_obj.batch_size)
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=(['sparse_categorical_accuracy']))
  model.summary()

  # Load and preprocess datasets
  train_dataset = imagenet_preprocessing.input_fn(is_training=True,
                                                  data_dir=flags_obj.data_dir,
                                                  batch_size=flags_obj.batch_size,
                                                  shuffle_seed = 42,
                                                  drop_remainder=True)
  validation_dataset = imagenet_preprocessing.input_fn(is_training=False,
                                                       data_dir=flags_obj.data_dir,
                                                       batch_size=flags_obj.batch_size,
                                                       shuffle_seed = 42,
                                                       drop_remainder=True)

  callbacks = []
  if flags_obj.enable_tensorboard:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=flags_obj.model_dir,
                                                    profile_batch=2))

  if flags_obj.enable_checkpoint_and_export:
    if flags_obj.model_dir is not None:
      ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
      callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))

  logging.info("Start training")
  history = model.fit(train_dataset,
                      epochs=flags_obj.train_epochs,
                      callbacks=callbacks,
                      validation_data=validation_dataset,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=1)
  logging.info("Train history: {}".format(history.history))

  stats = {}
  eval_output = model.evaluate(datasets['validation'],
                               steps=num_val_steps,
                               verbose=1)
  stats = tf_common.build_stats(history, eval_output, callbacks)

if __name__ == '__main__':
  common.define_keras_flags()
  app.run(main)
