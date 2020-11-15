# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Misc for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order

from absl import flags
import tensorflow as tf

from official.nlp.transformer import model_params
from official.nlp.transformer import misc
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

FLAGS = flags.FLAGS

PARAMS_MAP = {
    'tiny': model_params.TINY_PARAMS,
    'base': model_params.BASE_PARAMS,
    'big': model_params.BIG_PARAMS,
}


def define_transformer_flags():
  # add common flags
  misc.define_transformer_flags()
  flags.DEFINE_integer(
    name='train_epochs',
    default=1,
    help=flags_core.help_wrap(
        'Number of training epochs'))
  flags.DEFINE_integer(
    name='vocab_size',
    default=33786,
    help=flags_core.help_wrap(
        'Number of tokens generated when running `transformer/data_download.py`'))

