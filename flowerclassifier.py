import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random as rn
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Flowerclassifier():
  def __init__(self, client_id, data_dir, seed, save_checkpoint, load_checkpoint):
    #define preprocessing parameters
    #np.random.seed(37)
    rn.seed(1254)
    tf.random.set_seed(seed)
    self.client_id = client_id
    self.model_path = './bestmodel.hdf5'
    self.batch_size = 32
    self.img_height = 180
    self.img_width = 180
    self.num_classes = 5
    self.save_checkpoint = save_checkpoint
    self.load_checkpoint = load_checkpoint
    self.callbacks_list = None
    self.model = self.generate_model()
    self.train_ds, self.val_ds = self.get_datasets(data_dir)
    

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
    layers.Dropout(0.2), #nachlesen, Frochte
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


  def get_datasets(self, data_dir):
    #load training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.1,
      subset="training",
      seed=123,
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.1,
      subset="validation",
      seed=123,
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)

    #find class names
    self.class_names = train_ds.class_names
    print(self.class_names)

    return (train_ds, val_ds)


  def train_epoch(self, inner_epoch):
    history = self.model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=inner_epoch,
      callbacks= self.callbacks_list
    )
    return history


  def show_plot(self, history, epochs):
    #visualize training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


  def classify_single_picture(self):
    img = keras.preprocessing.image.load_img(
        'Flowers/myroses2.jpeg', target_size=(self.img_height, self.img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(predictions)
    print(score)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(self.class_names[np.argmax(score)], 100 * np.max(score))
    )








# usefull stuff for later

# split0, split1, split2 = tfds.even_splits('train', n=3)

#Checkpoint for best model
# filepath = './bestmodel.hdf5'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

#load old model to keep training
#model.load_weights('./savedmodel.hdf5')

# load_model = False #if this is True the model will be loaded and no training will be done
# if(load_model):
#   # load best model weights
#   model.load_weights(filepath)
#   jj = model.get_weights()
# else:
#   #train the model
#   epochs=3
#   history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs,
#     callbacks= callbacks_list
#   )

  # #visualize training results
  # acc = history.history['accuracy']
  # val_acc = history.history['val_accuracy']

  # loss = history.history['loss']
  # val_loss = history.history['val_loss']

  # epochs_range = range(epochs)

  # plt.figure(figsize=(8, 8))
  # plt.subplot(1, 2, 1)
  # plt.plot(epochs_range, acc, label='Training Accuracy')
  # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  # plt.legend(loc='lower right')
  # plt.title('Training and Validation Accuracy')

  # plt.subplot(1, 2, 2)
  # plt.plot(epochs_range, loss, label='Training Loss')
  # plt.plot(epochs_range, val_loss, label='Validation Loss')
  # plt.legend(loc='upper right')
  # plt.title('Training and Validation Loss')
  # plt.show()

#predict on new data
#example image
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
#
# img = keras.preprocessing.image.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
#
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
#
# #print(predictions)
# #print(score)
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )