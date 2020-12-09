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
import os

from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core

FLAGS = flags.FLAGS
BASE_LEARNING_RATE = 0.1  # This matches Jing's version.
TRAIN_TOP_1 = 'training_accuracy_top_1'
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
                (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
              ]


class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self,
               batch_size,
               epoch_size,
               warmup_epochs,
               boundaries,
               multipliers,
               compute_lr_on_cpu=True,
               name=None):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
      raise ValueError('The length of boundaries must be 1 less than the '
                       'length of multipliers')

    base_lr_batch_size = 256
    steps_per_epoch = epoch_size // batch_size

    self.rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self.step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
    self.lr_values = [self.rescaled_lr * m for m in multipliers]
    self.warmup_steps = warmup_epochs * steps_per_epoch
    self.compute_lr_on_cpu = compute_lr_on_cpu
    self.name = name

    self.learning_rate_ops_cache = {}

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    """Compute learning rate at given step."""
    with tf.name_scope('PiecewiseConstantDecayWithWarmup'):

      def warmup_lr(step):
        return self.rescaled_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))

      def piecewise_lr(step):
        return tf.compat.v1.train.piecewise_constant(step, self.step_boundaries,
                                                     self.lr_values)

      return tf.cond(step < self.warmup_steps, lambda: warmup_lr(step),
                     lambda: piecewise_lr(step))

  def get_config(self):
    return {
        'rescaled_lr': self.rescaled_lr,
        'step_boundaries': self.step_boundaries,
        'lr_values': self.lr_values,
        'warmup_steps': self.warmup_steps,
        'compute_lr_on_cpu': self.compute_lr_on_cpu,
        'name': self.name
    }


def define_keras_flags(dynamic_loss_scale=True,
                       model=False,
                       optimizer=False,
                       pretrained_filepath=False):
  """Define flags for Keras models."""
  flags_core.define_base(
      clean=True,
      num_gpu=False,
      run_eagerly=False,
      train_epochs=True,
      epochs_between_evals=True,
      distribution_strategy=False)
  flags_core.define_performance(
      num_parallel_calls=False,
      synthetic_data=False,
      dtype=False,
      all_reduce_alg=False,
      num_packs=False,
      tf_gpu_thread_mode=False,
      datasets_num_private_threads=True,
      dynamic_loss_scale=False,
      loss_scale=False,
      fp16_implementation=False,
      tf_data_experimental_slack=False,
      enable_xla=False)
#  flags_core.define_image()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_boolean(
      name='report_accuracy_metrics',
      default=True,
      help='Report metrics during training and evaluation.')
  flags.DEFINE_boolean(
      name='enable_tensorboard',
      default=False,
      help='Whether to enable Tensorborad.')
  flags.DEFINE_boolean(
      name='enable_checkpoint_and_export',
      default=False,
      help='Whether to enable a checkpoint callback and export the savedmodel.')

