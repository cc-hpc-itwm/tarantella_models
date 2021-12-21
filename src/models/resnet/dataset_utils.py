import tensorflow as tf

def get_tnt_cifar10_dataset(train_size, val_size, test_size, batch_size):
  # Load CIFAR10 dataset
  (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar10.load_data()

  # Reduce pixel values
  x_train_all, x_test_all = x_train_all / 255.0, x_test_all / 255.0

  # flatten the label values
  y_train_all, y_test_all = y_train_all.flatten(), y_test_all.flatten()

  # Training dataset
  x_train = x_train_all[:train_size]
  y_train = y_train_all[:train_size]

  # Validation dataset
  x_val = x_train_all[train_size:train_size+val_size]
  y_val = y_train_all[train_size:train_size+val_size]

  # Test dataset
  x_test = x_test_all[:test_size]
  y_test = y_test_all[:test_size]

  # Reshape
  x_train = x_train.reshape(train_size, 32, 32, 3).astype('float32') / 255
  x_val   = x_val.reshape(val_size, 32, 32, 3).astype('float32') / 255
  x_test  = x_test.reshape(test_size, 32, 32, 3).astype('float32') / 255
  y_train = y_train.astype('float32')
  y_val   = y_val.astype('float32')
  y_test  = y_test.astype('float32')

  # Create tf datasets
  train_dataset_raw = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  val_dataset_raw   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  test_dataset_raw  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

  # Shuffle dataset
  train_dataset = train_dataset_raw.shuffle(len(x_train), reshuffle_each_iteration=False)
  val_dataset   = val_dataset_raw.shuffle(len(x_val), reshuffle_each_iteration=False)
  test_dataset  = test_dataset_raw.shuffle(len(x_test), reshuffle_each_iteration=False)

  # Batch datasets based on mini batch size
  tnt_train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  tnt_val_dataset   = val_dataset.batch(batch_size, drop_remainder=True)
  tnt_test_dataset  = val_dataset.batch(batch_size, drop_remainder=True)

  return (tnt_train_dataset, tnt_val_dataset, tnt_test_dataset)
