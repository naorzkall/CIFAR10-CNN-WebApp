# CIFAR-10 Image Classification App

This repository contains a small web application for classifying CIFAR-10 images using a Convolutional Neural Network (CNN) model. The backend is implemented using Flask, and the frontend is built with Streamlit.

## Repository Contents

- `CIFAR-10-v3.ipynb`: Jupyter Notebook used to train the CNN model on the CIFAR-10 dataset.
- `flaskapi.py`: Flask API for serving the CNN model.
- `streamlitapp.py`: Streamlit application for interacting with the model.

## Dataset

The dataset used for training the CNN model is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. You can find more details about the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Analysis Overview

### Jupyter Notebook

The Jupyter Notebook (`CIFAR-10-v3.ipynb`) performs the following steps:
1. **Data Loading**: Importing the CIFAR-10 dataset.
2. **Data Preprocessing**: Normalizing the images and performing data augmentation.
3. **Model Building**: Constructing a CNN for image classification.
4. **Model Training**: Training the CNN model on the CIFAR-10 dataset.
5. **Model Evaluation**: Evaluating the performance of the model using appropriate metrics.
6. **Model Saving**: Saving the trained model for later use in the Flask API.

### Flask API

The Flask API (`flaskapi.py`) performs the following steps:
1. **Loading the Model**: Loading the saved CNN model.
2. **Receiving Requests**: Handling image classification requests.
3. **Preprocessing**: Preprocessing the input image.
4. **Prediction**: Using the model to predict the class of the image.
5. **Sending Response**: Sending the prediction result as a JSON response.

### Flask API Code

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
loaded_model = tf.keras.models.load_model('cnn_model_v3.keras')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_data = file.read()
    
    # Use IMREAD_COLOR to read the image in color  
    testing_image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)

    # Resize the image to 32x32
    resized_image = cv2.resize(testing_image, (32, 32))

    # Reshape the image to match the model's input shape
    reshaped_image = resized_image.reshape(-1, 32, 32, 3)
    reshaped_image = cv2.cvtColor(reshaped_image, cv2.COLOR_BGR2RGB)
    arr_image = np.array(reshaped_image, dtype='float32')

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    prediction = loaded_model.predict(arr_image)  
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = labels[predicted_class_index]
    encoded_image_data = base64.b64encode(testing_image).decode('utf-8')

    response_json = {'image_data': encoded_image_data, 'predicted_class_index': int(predicted_class_index), 'predicted_class_name': predicted_class_name}
    return jsonify(response_json)

if __name__ == '__main__':
    app.run(debug=True)
```

### Streamlit Application

The Streamlit application (`streamlitapp.py`) performs the following steps:
1. **User Interface**: Providing an interface for users to upload images.
2. **Sending Requests**: Sending the uploaded image to the Flask API.
3. **Displaying Results**: Displaying the classification results returned by the Flask API.

### Streamlit Application Code

```python
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
```

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Flask
- Streamlit
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cifar10-image-classification-app.git
    cd cifar10-image-classification-app
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow flask streamlit numpy opencv-python matplotlib requests
    ```

### Usage

#### Jupyter Notebook

1. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

2. Open and run the `CIFAR-10-v3.ipynb` notebook to train and save the CNN model.

#### Flask API

1. Run the Flask API:
    ```bash
    python flaskapi.py
    ```

#### Streamlit Application

1. Run the Streamlit app:
    ```bash
    streamlit run streamlitapp.py
    ```

2. Open the provided URL in your web browser to interact with the application.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The CIFAR-10 dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
- The open-source community for the tools and libraries used in this analysis.
```

Replace `yourusername` with your actual GitHub username, and make sure the `LICENSE` file is included in your repository if you're referencing it. This `README.md` provides a comprehensive overview of the project, instructions for setting up the environment, and guidelines for usage.
