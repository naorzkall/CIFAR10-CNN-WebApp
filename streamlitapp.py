import streamlit as st
import requests
import cv2
import numpy as np
import json
import base64
# Set page config
st.set_page_config(page_title='CIFAR-10 CNN Image Classifier', layout='wide')

# Title and description
st.title('THE CIFAR-10 CNN Image Classifier')
st.markdown('Upload an image for one of the following classes [airplane - automobile - bird - cat - deer - dog - frog - horse - ship - truck] and the CNN will predict its class.')

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
def predict_image_class(image):
    if uploaded_file is not None:
        image_bytes = image.read()

        data = {'image': image_bytes}

        response = requests.post("http://127.0.0.1:5000/predict", files=data)

        try:
            predicted_class = response.json()['predicted_class_name']
   
            return predicted_class
        except (KeyError, requests.exceptions.JSONDecodeError):
            st.subheader("Error: Failed to decode response from API. Check the API's response format.")
            return None

    else:
        if uploaded_file is None:
            st.subheader("Please upload an image file.")
        else:
            st.subheader("Please upload a valid image file (JPG, JPEG, or PNG).")
        return None

if st.button('Predict Class'):
    if uploaded_file is not None:
        # Display the prediction
        st.title('This image belongs to class:')
        st.title(predict_image_class(uploaded_file))
        # Display the uploaded image
        st.image(uploaded_file)
    else:
        st.warning("Please upload an image file.")

