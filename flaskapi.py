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
    reshaped_image=cv2.cvtColor(reshaped_image,cv2.COLOR_BGR2RGB)
    arr_image = np.array(reshaped_image, dtype='float32')

    
    labels = ["airplane","automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
    prediction = loaded_model.predict(arr_image)  
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = labels[predicted_class_index]
    encoded_image_data = base64.b64encode(testing_image).decode('utf-8')

    response_json = {'image_data': encoded_image_data, 'predicted_class_index':int(predicted_class_index),'predicted_class_name':predicted_class_name}
    return jsonify(response_json)


if __name__ == '__main__':
    app.run(debug=True)

