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

import os
import tempfile
# Import libraries
from absl import app
from absl import flags
from absl import logging
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

import tarantella

INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1



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
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["nepochs"] = flags_obj.train_epochs
    params["num_gpus"] = num_gpus
    params["data_dir"] = flags_obj.data_dir
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["max_io_parallelism"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals
    params["save_weights_only"] = flags_obj.save_weights_only

    logging.info("Running transformer with num_gpus = %d", num_gpus)
    self.internal_model = transformer.Transformer(params, name="transformer_v2")

    self.have_datapar = False
    if not flags_obj.without_datapar:
      tarantella.init(params["num_gpus"])
    self.have_datapar = True

    self.rank = 0
    self.comm_size = 1
    if self.have_datapar:
      self.rank = tarantella.get_rank()
      self.comm_size = tarantella.get_size()


  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    
    model = create_model(self.internal_model, self.params, is_train=True)
    if self.have_datapar:
      model = tarantella.model.TarantellaModel(model)
    
    opt = self._create_optimizer()
    model.compile(opt)
    model.summary()

    train_ds = data_pipeline.train_input_fn(params, self.comm_size, self.rank)
    map_data_fn = data_pipeline.map_data_for_transformer_fn
    train_ds = train_ds.map(
        map_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    callbacks = []
    if self.rank == 0:
      callbacks = misc.get_callbacks()

    cased_score, uncased_score = None, None
    cased_score_history, uncased_score_history = [], []

    logging.info("Start train")
    history = model.fit(train_ds,
                        epochs=self.params["nepochs"],
                        callbacks=callbacks,
                        verbose=(1 if self.rank == 0 else 0)
                        )
    logging.info("Train history: {}".format(history.history))

    logging.info("Start evaluation")
    if self.rank == 0:
      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        uncased_score, cased_score = self.eval()
        cased_score_history.append([current_iteration + 1, cased_score])
        uncased_score_history.append([current_iteration + 1, uncased_score])

    stats = ({
              "loss": train_loss
              } if history is None else {})

    if self.rank == 0:
      misc.update_stats(history, stats, callbacks)
      if uncased_score and cased_score:
        stats["bleu_uncased"] = uncased_score
        stats["bleu_cased"] = cased_score
        stats["bleu_uncased_history"] = uncased_score_history
        stats["bleu_cased_history"] = cased_score_history
    return stats

  def eval(self):
    """Evaluates the model."""
    if not self.predict_model:
      self.predict_model = create_model(self.internal_model, self.params, False)
    self.predict_model.summary()
    return transormer_main.evaluate_and_log_bleu(
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
  task = TransformerTntTask(flags_obj)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)

  if flags_obj.mode == "train":
    task.train()
  elif flags_obj.mode == "predict":
    task.predict()
  elif flags_obj.mode == "eval":
    task.eval()
  else:
    raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tnt_misc.define_transformer_flags()
  app.run(main)


