from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize

app = Flask(__name__)

# Load the trained model from the specified path
model_path = r"D:\d\Imageclassifier\svm_model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define categories (must match those used during training)
categories = ['empty', 'not_empty']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error='No selected file')

    if file:
        # Process the uploaded image
        img = imread(file)
        img = resize(img, (15, 15))
        img = img.flatten().reshape(1, -1)

        # Predict using the model
        prediction = model.predict(img)
        predicted_class = categories[prediction[0]]

        return render_template('result.html', prediction=predicted_class)

    return render_template('result.html', error='Unable to process the file')

if __name__ == '__main__':
    app.run(debug=True)
