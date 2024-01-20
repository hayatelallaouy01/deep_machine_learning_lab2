# Lab22 : Classification des images fruits & légumes CNN
# Réalisé par : Hayat el allaouy /Emsi 2023-2024
# Réference : https://colab.research.google.com/drive/1lTjZ22a_T5jMOJtcNX2_AvZoAKrtZMVv#scrollTo=8lCi3O2Otp9D
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Step 1 : DataSet
# Data manipulation

img_height, img_width = 32, 32
batch_size = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/validation",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# Data visualisation

class_names = ["apple", "banana", "orange","cucumber","eggplant","garlic","grapes","kiwi","lemon","mango", "onion","paprika","pear","pineapple","pomegranate","spinach","sweetpotato","tomato","turnip","watermelon"]

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

# Step 2 : Model

model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(20)
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

# Step 3 : Train

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)

# Step 4 : Test

model.evaluate(test_ds)
import numpy
plt.figure(figsize=(10,10))
for images, labels in test_ds.take(1):
  classifications = model(images)
  # print(classifications)

  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    index = numpy.argmax(classifications[i])
    plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])
    plt.show()
# Save and deploy the tflite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", 'wb') as f:
  f.write(tflite_model)


