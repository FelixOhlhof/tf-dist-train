from cgi import test
import tensorflow as tf
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc

class Flowerclassifier():
  def __init__(self, data_dir, seed):
    #define preprocessing parameters
    rn.seed(1254)
    tf.random.set_seed(seed)
    self.batch_size = 32
    self.img_height = 180
    self.img_width = 180
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
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),
      # The second convolution
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # The third convolution
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # The fourth convolution
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # # The fifth convolution
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # Flatten the results to feed into a DNN
      tf.keras.layers.Flatten(),
      # 512 neuron hidden layer
      tf.keras.layers.Dense(512, activation='relu'),
      # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

    return model


  def get_datasets(self, data_dir):
    #load training data
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)
    train_dir = f"{data_dir}\\train"
    test_dir = f"{data_dir}\\test"
    train_classes = [ f.name for f in os.scandir(train_dir) if f.is_dir() ]
    test_classes = [ f.name for f in os.scandir(test_dir) if f.is_dir() ]

    # Flow training images in batches of 120 using train_datagen generator
    train_ds = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            classes = train_classes,
            target_size=(self.img_height, self.img_width),  # All images will be resized to 200x200
            batch_size=120,
            # Use binary labels
            class_mode='binary')

    # Flow validation images in batches of 19 using valid_datagen generator
    val_ds = validation_datagen.flow_from_directory(
            test_dir,  # This is the source directory for training images
            classes = test_classes,
            target_size=(self.img_height, self.img_width),  # All images will be resized to 200x200
            batch_size=19,
            # Use binary labels
            class_mode='binary',
            shuffle=False)

    return (train_ds, val_ds)


  def train_epoch(self, epochs):
    history = self.model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=epochs,
      callbacks= None
    )

    #ROC AUC
    self.model.evaluate(self.val_ds)
    preds = self.model.predict(self.val_ds, verbose=1)
    fpr, tpr, _ = roc_curve(self.val_ds.classes, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return history