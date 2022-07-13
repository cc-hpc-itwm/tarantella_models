import argparse
import os
import sys

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
#import visualkeras

import tensorflow as tf

cnn_models = {'resnet50': tf.keras.applications.resnet50.ResNet50,
              'resnet101': tf.keras.applications.resnet.ResNet101,
              'resnet152': tf.keras.applications.resnet.ResNet152,
              'efficientnetV2B0': tf.keras.applications.efficientnet_v2.EfficientNetV2B0,
              'efficientnetV2B1': tf.keras.applications.efficientnet_v2.EfficientNetV2B1,
              'efficientnetV2B2': tf.keras.applications.efficientnet_v2.EfficientNetV2B2,
              'efficientnetV2B3': tf.keras.applications.efficientnet_v2.EfficientNetV2B3,
              'efficientnetV2S': tf.keras.applications.efficientnet_v2.EfficientNetV2S,
              'efficientnetV2M': tf.keras.applications.efficientnet_v2.EfficientNetV2M,
              'efficientnetV2L': tf.keras.applications.efficientnet_v2.EfficientNetV2L,
              }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--profile_dir", help="directory for profiles")
  parser.add_argument("--model_arch", type=str, default="resnet50",
                      help = f"Choose one of: {list(cnn_models.keys())}")

  args = parser.parse_args()
  if not args.model_arch in list(cnn_models.keys()):
    sys.exit(f"ERROR: Model `{args.model_arch}` not supported (choose one of: {list(cnn_models.keys())})")
  return args  

args = parse_args()

def get_reference_compile_params():
  return {'optimizer' : tf.keras.optimizers.SGD(learning_rate=0.1,
                                                momentum=0.9),
          'loss' : tf.keras.losses.SparseCategoricalCrossentropy(),
          'metrics' : [tf.keras.metrics.SparseCategoricalAccuracy()]}

def print_flops(model):
  forward_pass = tf.function(
      model.call,
      input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

  options=ProfileOptionBuilder.float_operation()
  options['order_by'] = ''
#  options['output'] = f"file=flops_{model.name}.txt"
  graph_info = profile(forward_pass.get_concrete_function().graph,
                          options=options)

  flops = graph_info.total_float_ops
  print('Flops: {:,}'.format(flops))

if __name__ == '__main__':
  model_arch = cnn_models[args.model_arch]

  model = model_arch(include_top=True,
                      weights=None,
                      classes=1000,
                      input_shape=(224, 224, 3),
                      input_tensor=None,
                      pooling=None,
                      classifier_activation='softmax')
  model.compile(**get_reference_compile_params())
  model.summary()
  
  print_flops(model)
  #visualkeras.layered_view(model, legend = True,  to_file=f"{model.name}.png")