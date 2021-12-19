import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _gen_l2_regularizer(use_l2_regularizer=True,
                        l2_weight_decay=1e-4):
  return tf.keras.regularizers.L2(l2_weight_decay) if use_l2_regularizer else None

def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):

  filters1, filters2 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1,
                    kernel_size,
                    padding='same',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=batch_norm_decay,
                                epsilon=batch_norm_epsilon,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2,
                    kernel_size,
                    padding='same',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                    name=conv_name_base + '2b')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=batch_norm_decay,
                                epsilon=batch_norm_epsilon,
                                name=bn_name_base + '2b')(x)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):

  filters1, filters2 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1,
                    kernel_size,
                    strides=strides,
                    padding='same',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=batch_norm_decay,
                                epsilon=batch_norm_epsilon,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2,
                    kernel_size,
                    padding='same',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                    name=conv_name_base + '2b')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=batch_norm_decay,
                                epsilon=batch_norm_epsilon,
                                name=bn_name_base + '2b')(x)

  shortcut = layers.Conv2D(filters2,
                           (1, 1),
                           strides=strides,
                           use_bias=False,
                           kernel_initializer='he_normal',
                           kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                           name=conv_name_base + '1')(input_tensor)
  shortcut = layers.BatchNormalization(axis=bn_axis,
                                       momentum=batch_norm_decay,
                                       epsilon=batch_norm_epsilon,
                                       name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x

def resnet32(num_classes,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):

  input_shape = (32, 32, 3)
  img_input = layers.Input(shape=input_shape, name='Inputs')

  if tf.keras.backend.image_data_format() == 'channels_first':
    x = layers.Permute((3, 1, 2))(x)
    bn_axis = 1
  else:  # channels_last
    x = img_input
    bn_axis = 3

  block_config = dict(use_l2_regularizer=use_l2_regularizer,
                      batch_norm_decay=batch_norm_decay,
                      batch_norm_epsilon=batch_norm_epsilon)

  x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
  x = layers.Conv2D(16,
                    (3, 3),
                    strides=(1, 1),
                    padding='valid',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                    name='conv1')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=batch_norm_decay,
                                epsilon=batch_norm_epsilon,
                                name='bn_conv1')(x)
  x = layers.Activation('relu')(x)

  x = conv_block(x, 3, [16, 16], stage=2, block='a', strides=(1, 1), **block_config)
  x = identity_block(x, 3, [16, 16], stage=2, block='b', **block_config)
  x = identity_block(x, 3, [16, 16], stage=2, block='c', **block_config)
  x = identity_block(x, 3, [16, 16], stage=2, block='d', **block_config)
  x = identity_block(x, 3, [16, 16], stage=2, block='e', **block_config)

  x = conv_block(x, 3, [32, 32], stage=3, block='a', strides=(2, 2), **block_config)
  x = identity_block(x, 3, [32, 32], stage=3, block='b', **block_config)
  x = identity_block(x, 3, [32, 32], stage=3, block='c', **block_config)
  x = identity_block(x, 3, [32, 32], stage=3, block='d', **block_config)
  x = identity_block(x, 3, [32, 32], stage=3, block='e', **block_config)

  x = conv_block(x, 3, [64, 64], stage=4, block='a', strides=(2, 2), **block_config)
  x = identity_block(x, 3, [64, 64], stage=4, block='b', **block_config)
  x = identity_block(x, 3, [64, 64], stage=4, block='c', **block_config)
  x = identity_block(x, 3, [64, 64], stage=4, block='d', **block_config)
  x = identity_block(x, 3, [64, 64], stage=4, block='e', **block_config)

  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(num_classes,
                   kernel_initializer=tf.compat.v1.keras.initializers.random_normal(stddev=0.01),
                   kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                   bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                   name='fc10')(x)

  x = layers.Activation('softmax', dtype='float32', name='Outputs')(x)

  return tf.keras.Model(img_input, x, name='ResNet32')