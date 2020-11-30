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
    'base': model_params.BASE_MULTI_GPU_PARAMS,
    'big': model_params.BIG_MULTI_GPU_PARAMS,
}

def get_model_params(param_set):
  """Gets predefined model params."""
  if param_set not in PARAMS_MAP:
    raise ValueError('Not valid params: param_set={}'.format(param_set))
  return PARAMS_MAP[param_set].copy()

def define_transformer_flags():
  # add common flags
  flags_core.define_base(data_dir=True, model_dir=True, clean=False, train_epochs=True,
                         epochs_between_evals=True, stop_threshold=False,
                         batch_size=True, num_gpu=False, hooks=False, export_dir=False,
                         distribution_strategy=False, run_eagerly=False)

  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=False,
      loss_scale=True,
      all_reduce_alg=False,
      num_packs=False,
      tf_gpu_thread_mode=True,
      datasets_num_private_threads=True,
      enable_xla=False,
      fp16_implementation=False
  )

  flags_core.define_benchmark()

  flags.DEFINE_boolean(
      name='enable_time_history', default=True,
      help='Whether to enable TimeHistory callback.')

  flags.DEFINE_boolean(
      name='enable_tensorboard', default=False,
      help='Whether to enable Tensorboard callback.')

  flags.DEFINE_boolean(
      name='enable_metrics_in_training', default=False,
      help='Whether to enable metrics during training.')

  # Set flags from the flags_core module as 'key flags' so they're listed when
  # the '-h' flag is used. Without this line, the flags defined above are
  # only shown in the full `--helpful` help text.
  flags.adopt_module_key_flags(flags_core)

  # Add transformer-specific flags
  flags.DEFINE_enum(
      name='param_set', short_name='mp', default='big',
      enum_values=PARAMS_MAP.keys(),
      help=flags_core.help_wrap(
          'Parameter set to use when creating and training the model. The '
          'parameters define the input shape (batch size and max length), '
          'model configuration (size of embedding, # of hidden layers, etc.), '
          'and various other settings. The big parameter set increases the '
          'default batch size, embedding/hidden size, and filter size. For a '
          'complete list of parameters, please see model/model_params.py.'))

  flags.DEFINE_integer(
      name='max_length', short_name='ml', default=256,
      help=flags_core.help_wrap(
          'Max sentence length for Transformer. Default is 256. Note: Usually '
          'it is more effective to use a smaller max length if static_batch is '
          'enabled, e.g. 64.'))

  # BLEU score computation
  flags.DEFINE_string(
      name='bleu_source', short_name='bls', default=None,
      help=flags_core.help_wrap(
          'Path to source file containing text translate when calculating the '
          'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
          ))
  flags.DEFINE_string(
      name='bleu_ref', short_name='blr', default=None,
      help=flags_core.help_wrap(
          'Path to source file containing text translate when calculating the '
          'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
          ))
  flags.DEFINE_string(
      name='vocab_file', short_name='vf', default=None,
      help=flags_core.help_wrap(
          'Path to subtoken vocabulary file. If data_download.py was used to '
          'download and encode the training data, look in the data_dir to find '
          'the vocab file.'))

  flags.DEFINE_integer(
      name='decode_batch_size',
      default=32,
      help=flags_core.help_wrap(
          'Global batch size used for Transformer autoregressive decoding on '
          'TPU.'))
  flags.DEFINE_integer(
      name='decode_max_length',
      default=97,
      help=flags_core.help_wrap(
          'Max sequence length of the decode/eval data. This is used by '
          'Transformer autoregressive decoding on TPU to have minimum '
          'paddings.'))

  flags.DEFINE_bool(
      name='enable_checkpointing',
      default=True,
      help=flags_core.help_wrap(
          'Whether to do checkpointing during training. When running under '
          'benchmark harness, we will avoid checkpointing.'))

  flags_core.set_defaults(data_dir='/tmp/translate_ende',
                          model_dir='/tmp/transformer_model',
                          batch_size=None)

  # pylint: disable=unused-variable
  @flags.multi_flags_validator(
      ['bleu_source', 'bleu_ref'],
      message='Both or neither --bleu_source and --bleu_ref must be defined.')
  def _check_bleu_files(flags_dict):
    return (flags_dict['bleu_source'] is None) == (
        flags_dict['bleu_ref'] is None)

  @flags.multi_flags_validator(
      ['bleu_source', 'bleu_ref', 'vocab_file'],
      message='--vocab_file must be defined if --bleu_source and --bleu_ref '
              'are defined.')
  def _check_bleu_vocab_file(flags_dict):
    if flags_dict['bleu_source'] and flags_dict['bleu_ref']:
      return flags_dict['vocab_file'] is not None
    return True
  # pylint: enable=unused-variable

  flags.DEFINE_integer(
      name='steps_per_epoch', short_name='spe', default=10000,
      help=flags_core.help_wrap('The number of steps used to train.'))
