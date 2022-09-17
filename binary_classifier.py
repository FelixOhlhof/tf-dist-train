from cgi import test
import tensorflow as tf
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import os
from datetime import datetime
from keras.preprocessing import image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from PIL import Image
import numpy as np
from skimage import transform

class BinaryClassifier():
  def __init__(self, data_dir, seed, client_id):
    #define preprocessing parameters
    rn.seed(1254)
    tf.random.set_seed(seed)
    self.batch_size = 32
    self.img_height = 180
    self.img_width = 180
    self.client_id = client_id
    self.class_names = []
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

    self.class_names = train_classes

    # Flow training images in batches of 120 using train_datagen generator
    train_ds = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            classes = train_classes,
            target_size=(self.img_height, self.img_width),  # All images will be resized
            batch_size=120,
            # Use binary labels
            class_mode='binary')

    # Flow validation images in batches of 19 using valid_datagen generator
    val_ds = validation_datagen.flow_from_directory(
            test_dir,  # This is the source directory for training images
            classes = test_classes,
            target_size=(self.img_height, self.img_width),  # All images will be resized
            batch_size=19,
            # Use binary labels
            class_mode='binary',
            shuffle=False)

    return (train_ds, val_ds)

  def train_epoch(self, epochs):
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    history = self.model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=epochs,
      callbacks=None
    )
    self.model.evaluate(self.val_ds)
    return history

  def save_roc_curve(self, test_id, client_id, strategy):
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

      now = datetime.now().strftime("%H.%M.%S")
      graph = f"results\\{strategy}\\graphs\\Test_{test_id}_Worker_{client_id}_ROC.png"
      mng = plt.get_current_fig_manager()
      mng.full_screen_toggle()
      plt.savefig(graph, dpi=600)
      plt.close()
      return graph
      # plt.show()
    
  def calculate_scores(self):
      # assign directory
      directory = 'Flowers/'
      scores = []
      
      # iterate over files in
      # that directory
      for root, dirs, files in os.walk(directory):
          for filename in files:
              # img = keras.preprocessing.image.load_img(
              #     os.path.join(root, filename), target_size=(self.img_height, self.img_width)
              # )
              img = keras.preprocessing.image.load_img(
                  r"C:\Users\felix\Desktop\1.png", target_size=(self.img_height, self.img_width)
              )
              img_array = keras.preprocessing.image.img_to_array(img)
              img_array = np.expand_dims(img_array, axis=0) # Create a batch
              predictions = self.model.predict(self.load_image(r"C:\Users\felix\Desktop\1.png"))
              
              score = tf.nn.softmax(predictions[0])
              print(
                  "{} belongs most likely to {} with a {:.2f} percent confidence."
                  .format(os.path.join(root, filename), self.class_names[np.argmax(score)], 100 * np.max(score))
              )
              scores.append(round(np.max(score), 2))
      return scores

  def load_image(self, img_path, show=False):

    img = keras.preprocessing.image.load_img(
                  img_path, target_size=(self.img_height, self.img_width)
              )
    img_tensor = keras.preprocessing.image.img_to_array(img)                   # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

  def save_model(self, test_id, client_id, strategy):
    model_path = f"results\\{strategy}\\models\\Test_{test_id}_Worker_{client_id}.h5"
    self.model.save(model_path)
    return model_path