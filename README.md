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

### Streamlit Application

The Streamlit application (`streamlitapp.py`) performs the following steps:
1. **User Interface**: Providing an interface for users to upload images.
2. **Sending Requests**: Sending the uploaded image to the Flask API.
3. **Displaying Results**: Displaying the classification results returned by the Flask API.


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
