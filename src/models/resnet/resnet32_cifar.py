from absl import app
from absl import flags
import logging
import os

import tensorflow as tf

import common
import dataset_utils
import resnet32 as resnet_model

from utils import RuntimeProfiler
# Enable Tarantella
import tarantella as tnt

def get_optimizer(batch_size):
  lr_schedule = common.PiecewiseConstantDecayWithWarmup(
                        batch_size=batch_size,
                        epoch_size=45000,
                        warmup_epochs=common.LR_SCHEDULE[0][1],
                        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
                        multipliers=list(p[0] for p in common.LR_SCHEDULE),
                        compute_lr_on_cpu=True)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
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
  (train_dataset, validation_dataset, _) = dataset_utils.get_tnt_cifar10_dataset(45000, 5000, 10000, batch_size)

  # Create model and wrap it into a Tarantella model
  model = resnet_model.resnet32(num_classes=10)
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
                      verbose=flags_obj.verbose,
                      **kwargs)
  logging.info("Train history: {}".format(history.history))

  kwargs = {'tnt_distribute_dataset': flags_obj.auto_distributed}
  eval_output = model.evaluate(validation_dataset,
                               verbose=flags_obj.verbose,
                               **kwargs)
  

if __name__ == '__main__':
  common.define_keras_flags()
  app.run(main)
