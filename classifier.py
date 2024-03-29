import tensorflow as tf
import numpy as np
import random as rn
import os
import re, time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Classifier():
  def __init__(self, client_id, client_count, train_dir, val_dir, seed, save_checkpoint, load_checkpoint, batch_size, shuffle_data_mode):
    #define preprocessing parameters
    np.random.seed(37)
    rn.seed(seed)
    tf.random.set_seed(seed)
    self.seed = seed
    self.client_id = client_id
    self.client_count = client_count
    self.model_path = './bestmodel.hdf5'
    self.batch_size = batch_size
    self.img_height = 180
    self.img_width = 180
    self.num_classes = 5
    self.save_checkpoint = save_checkpoint
    self.load_checkpoint = load_checkpoint
    self.callbacks_list = None
    self.model = self.generate_model()
    self.data = []
    self.train_dir = train_dir
    self.val_dir = val_dir
    
    if(shuffle_data_mode):
      self.data = self.get_datasets_in_shuffle_mode(train_dir)
    else:
      self.data.append(self.get_datasets(train_dir))


  def generate_model(self):
    #create the model
    data_augmentation = keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal",
                                                    input_shape=(self.img_height,
                                                                self.img_width,
                                                                3)),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.RandomZoom(0.1),
    ]
    )

    model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(self.num_classes)
    ])

    #compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    if(self.save_checkpoint and self.client_id == 1):
      #Checkpoint for best model, only for 1 client
      filepath = self.model_path
      checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
      self.callbacks_list = [checkpoint]

    if(self.load_checkpoint and os.path.exists(self.model_path)):
      #load old model to keep training
      model.load_weights(self.model_path)
      print("Loaded saved model!")

    return model


  def get_datasets_in_shuffle_mode(self, train_dir):
    data = []

    for i in range(1, self.client_count + 1):
      data.append(self.get_datasets(re.sub(r".$", str(i), train_dir)))

    return data


  def get_datasets(self, train_dir, validation_dir = None):
    if(validation_dir == None):
      #load training data
      train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

      val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)
    else:
      #load training data
      train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=None,
        subset=None,
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

      val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_dir,
        validation_split=None,
        subset=None,
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

    #find class names
    self.class_names = train_ds.class_names
    print(self.class_names)

    return ((train_ds, val_ds))


  def train_epoch(self, inner_epoch):
    history = self.model.fit(
      self.data[0][0],
      validation_data=self.data[0][1],
      epochs=inner_epoch,
      callbacks= self.callbacks_list
    )
    return history


  def train_epoch_in_shuffle_mode(self, inner_epoch, current_epoch):
    ds_index = (current_epoch + self.client_id - 1) % self.client_count
    print("Data shuffle: Continuing with slice ", ds_index + 1, "...")

    history = self.model.fit(
      self.data[ds_index][0],
      validation_data=self.data[ds_index][1],
      epochs=inner_epoch,
      callbacks= self.callbacks_list
    )
    return history


  def calculate_score(self):
    # assign directory
    directory = 'Flowers/'
    scores = []
    
    # iterate over files in
    # that directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            img = keras.preprocessing.image.load_img(
                os.path.join(root, filename), target_size=(self.img_height, self.img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            print(
                "{} belongs most likely to {} with a {:.2f} percent confidence."
                .format(os.path.join(root, filename), self.class_names[np.argmax(score)], 100 * np.max(score))
            )
            scores.append(np.max(score))
    return sum(scores) / len(scores)


  def evaluate(self, client_id):
    if(client_id != 1):
      return 0, 0, 0, 0, 0

    ds = self.get_datasets(self.train_dir, self.val_dir)

    eval_train = self.model.evaluate(ds[0])
    start_time = time.time()
    eval_test = self.model.evaluate(ds[1]) # measure time needed for evaluation validation ds
    end_time = time.time()
    time_lapsed = end_time - start_time
    return eval_train[1], eval_train[0], eval_test[1], eval_test[0], time_lapsed

