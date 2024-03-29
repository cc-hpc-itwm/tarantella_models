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

# Import libraries
from absl import app
from absl import flags
import tensorflow as tf

#import layer_helpers.einsum_dense 
#import layer_helpers.multi_head_attention

#tf.keras.layers.experimental.EinsumDense = layer_helpers.einsum_dense.EinsumDense
#tf.keras.layers.MultiHeadAttention = layer_helpers.multi_head_attention.MultiHeadAttention

from official.legacy.transformer import metrics
from official.legacy.transformer import misc
from official.legacy.transformer import optimizer
from official.legacy.transformer import transformer
from official.legacy.transformer import transformer_main
from official.utils.misc import keras_utils

import data_pipeline
import misc as tnt_misc
import models.utils as utils

import tarantella as tnt
  
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
    """
    self.flags_obj = flags_obj
    self.params = tnt_misc.get_model_params(flags_obj.param_set)
    self.params["train_epochs"] = flags_obj.train_epochs
    self.params["epochs_between_evals"] = flags_obj.epochs_between_evals
    self.params["num_sentences"] = flags_obj.num_sentences
    self.params["num_eval_sentences"] = flags_obj.num_eval_sentences
    self.params["batch_size"] = flags_obj.batch_size or self.params["default_batch_size"]

    self.params["synthetic_data"] = flags_obj.synthetic_data

    self.params["data_dir"] = flags_obj.data_dir
    self.params["vocab_size"] = flags_obj.vocab_size or self.params["vocab_size"]
    self.params["max_length"] = flags_obj.max_length
    self.params["decode_batch_size"] = flags_obj.decode_batch_size
    self.params["decode_max_length"] = flags_obj.decode_max_length
    self.params["max_io_parallelism"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    self.params["use_synthetic_data"] = flags_obj.use_synthetic_data
    self.params["dtype"] = tf.float32

    # Transformer model used both as Tarantella model (in training) and as a serial
    # model for inference
    internal_model = transformer.Transformer(self.params, name="transformer")

    # The train model includes an additional logits layer and a customized loss
    self.train_model = create_model(internal_model, self.params, is_train = True)
    # Enable distributed training
    self.train_model = tnt.Model(self.train_model,
                                 parallel_strategy = tnt.ParallelStrategy.DATA)

    # The inference model is wrapped as a different Keras model that does not use labels
    self.predict_model = create_model(internal_model, self.params, is_train = False)

  def train_and_eval(self):
    """Trains the model."""
    lr_schedule = optimizer.LearningRateSchedule(self.params["learning_rate"], self.params["hidden_size"],
                                                 self.params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(lr_schedule,
                                   self.params["optimizer_adam_beta1"],
                                   self.params["optimizer_adam_beta2"],
                                   epsilon=self.params["optimizer_adam_epsilon"])
    self.train_model.compile(opt)
    self.train_model.summary()

    # create train dataset
    num_ranks = None if self.flags_obj.auto_data_dist else tnt.get_size()
    rank = None if self.flags_obj.auto_data_dist else tnt.get_rank()
    train_ds = data_pipeline.train_input_fn(self.params,
                                            shuffle_seed = 42,
                                            num_ranks = num_ranks,
                                            rank = rank)

    # enable global callbacks
    callbacks = []
    if self.flags_obj.enable_tensorboard and self.flags_obj.model_dir:
      callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.flags_obj.model_dir))

    # enable logging callbacks only on the master rank
    if self.flags_obj.enable_profiler:
      profiler_callback = utils.RuntimeProfiler(batch_size = self.params["batch_size"],
                                                logging_freq = self.flags_obj.logging_freq,
                                                print_freq = self.flags_obj.print_freq)
      profiler_callback = tnt.keras.callbacks.Callback(profiler_callback,
                                                       run_on_all_ranks = False,
                                                       aggregate_logs = False)
      callbacks.append(profiler_callback)

    # print messages only once
    if tnt.is_master_rank():
      logging.info("Start train")

    stats = {}
    for epoch in range(0, self.params["train_epochs"], self.params["epochs_between_evals"]):
      # as our dataset is distributed manually, disable the automatic Tarantella distribution
      history = self.train_model.fit(train_ds,
                                     callbacks = callbacks,
                                     tnt_distribute_dataset = self.flags_obj.auto_data_dist,
                                     initial_epoch = epoch,
                                     epochs = epoch + min(self.params["epochs_between_evals"],
                                                          self.params["train_epochs"]-epoch),
                                     verbose = 2)

      if tnt.is_master_rank():
        logging.info("Train history: {}".format(history.history))

      # if tnt.is_master_rank():
      #   eval_stats = self.eval()
      #   stats.update(eval_stats)

    return stats

  def eval(self):
    """Evaluates the model."""
    stats = {}
    
    if not (self.flags_obj.bleu_source and self.flags_obj.bleu_ref):
      logging.info("No evaluation dataset provided. Skippping evaluation.")
      return stats

    logging.info("Start evaluation")
    uncased_score, cased_score = transformer_main.evaluate_and_log_bleu(
                                                self.predict_model,
                                                self.params,
                                                self.flags_obj.bleu_source,
                                                self.flags_obj.bleu_ref,
                                                self.flags_obj.vocab_file)
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
    return stats

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
  task.train_and_eval()

if __name__ == "__main__":
  tnt_misc.define_transformer_flags()
  app.run(main)


