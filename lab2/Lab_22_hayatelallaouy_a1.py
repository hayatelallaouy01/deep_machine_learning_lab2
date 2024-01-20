# Lab22 : Classification des images fruits & légumes CNN
# Réalisé par : Housna Aghzer /Emsi 2023-2024
# Réference : https://colab.research.google.com/drive/1lTjZ22a_T5jMOJtcNX2_AvZoAKrtZMVv#scrollTo=8lCi3O2Otp9D

import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names
class_names = ["apple", "banana", "capsicum", "carrot", "cauliflower", "corn", "cucumber", "eggplant", "grapes", "kiwi", "mango", "onion", "orange", "pear", "peas", "pineapple", "pomegranate", "potato", "tomato", "watermelon"]

st.set_page_config(page_title="Fruit and Vegetable Classifier", layout="wide", initial_sidebar_state="expanded")
st.title(' Classification of Fruits Vegetables')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"]) # Accept both JPEG and PNG images

if uploaded_file is not None:
 image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(32, 32))
 st.image(image, width=100) # Display the image with a width of 100 pixels

 image = tf.keras.preprocessing.image.img_to_array(image)
 image = np.expand_dims(image, axis=0)

 interpreter.set_tensor(input_details[0]['index'], image)
 interpreter.invoke()

 output_data = interpreter.get_tensor(output_details[0]['index'])
 predicted_class = np.argmax(output_data)
 st.write(f"Prediction {class_names[predicted_class]}") # Display the predicted class
