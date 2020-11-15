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
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile
# Import libraries
from absl import app
from absl import flags
import tensorflow as tf

import layer_helpers.einsum_dense 
import layer_helpers.multi_head_attention

tf.keras.layers.experimental.EinsumDense = layer_helpers.einsum_dense.EinsumDense
tf.keras.layers.MultiHeadAttention = layer_helpers.multi_head_attention.MultiHeadAttention

from official.common import distribute_utils
from official.modeling import performance
from official.nlp.transformer import metrics
from official.nlp.transformer import misc
from official.nlp.transformer import optimizer
from official.nlp.transformer import transformer
from official.nlp.transformer import transformer_main
from official.nlp.transformer import translate
from official.nlp.transformer.utils import tokenizer
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

import data_pipeline
import misc as tnt_misc

import tarantella as tnt

tnt.init()
  
def create_model(internal_model, params, is_train):
  """Creates transformer model."""
  with tf.name_scope("model"):
    if is_train:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
      
      logits = internal_model([inputs, targets], training=is_train)
      vocab_size = params["vocab_size"]
      label_smoothing = params["label_smoothing"]
      if params["enable_metrics_in_training"]:
        logits = metrics.MetricLayer(vocab_size)([logits, targets])
      logits = tf.keras.layers.Lambda(
          lambda x: x, name="logits", dtype=tf.float32)(
              logits)
      model = tf.keras.Model([inputs, targets], logits)
      loss = metrics.transformer_loss(logits, targets, label_smoothing,
                                      vocab_size)
      model.add_loss(loss)
      return model

    else:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      ret = internal_model([inputs], training=is_train)
      outputs, scores = ret["outputs"], ret["scores"]
      return tf.keras.Model(inputs, [outputs, scores])


class TransformerTntTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = misc.get_model_params(flags_obj.param_set, num_gpus)

    self.params["train_epochs"] = flags_obj.train_epochs
    self.params["batch_size"] = flags_obj.batch_size or self.params["default_batch_size"]

    self.params["data_dir"] = flags_obj.data_dir
    self.params["vocab_size"] = flags_obj.vocab_size or self.params["vocab_size"]
    self.params["max_length"] = flags_obj.max_length
    self.params["decode_batch_size"] = flags_obj.decode_batch_size
    self.params["decode_max_length"] = flags_obj.decode_max_length
    self.params["padded_decode"] = flags_obj.padded_decode
    self.params["max_io_parallelism"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    self.params["use_synthetic_data"] = flags_obj.use_synthetic_data
    self.params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    self.params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training

    self.internal_model = transformer.Transformer(self.params, name="transformer_v2")

  def train(self):
    """Trains the model."""
    flags_obj = self.flags_obj
    
    model = create_model(self.internal_model, self.params, is_train=True)
    model = tnt.Model(model)
    
    opt = self._create_optimizer()
    model.compile(opt)
    model.summary()

    train_ds = data_pipeline.train_input_fn(self.params)

    callbacks = []
    if self.flags_obj.enable_tensorboard:
      callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.flags_obj.model_dir))

    logging.info("Start train")
    history = model.fit(train_ds,
                        epochs=self.params["train_epochs"],
                        callbacks=callbacks,
                        verbose=1)
    logging.info("Train history: {}".format(history.history))

    stats = {}
    misc.update_stats(history, stats, callbacks)
    return stats

  def eval(self):
    """Evaluates the model."""
    logging.info("Start evaluation")
  
    if not self.predict_model:
      self.predict_model = create_model(self.internal_model, self.params, False)
    return transformer_main.evaluate_and_log_bleu(
        self.predict_model, self.params, self.flags_obj.bleu_source,
        self.flags_obj.bleu_ref, self.flags_obj.vocab_file)


  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(
        lr_schedule,
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    opt = performance.configure_optimizer(opt)
    return opt


def main(_):
  flags_obj = flags.FLAGS

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)

  task = TransformerTntTask(flags_obj)
  task.train()
  task.eval()

if __name__ == "__main__":
  tnt_misc.define_transformer_flags()
  app.run(main)


