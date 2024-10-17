# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

# Create the Flask application
app = Flask(__name__)

# Load the model
model = load_model('model/crash_car.h5')

# Set the upload folder path
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize a list to store crash data
crash_data = []  # Store crash prediction data

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# File processing page
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')

    if file:
        # If the file is an image or a video
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # If it's an image
            return process_image(filepath)
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # If it's a video
            return process_video(filepath, file.filename)

    return redirect(request.url)

def process_image(filepath):
    # Process the image
    img = plt.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Make a prediction
    prediction = model.predict(img_resized)
    predicted_label = (prediction > 0.5).astype(int)
    result = 'Crash' if predicted_label[0][0] == 1 else 'No Crash'

    # Store crash data
    crash_data.append(result)

    # Convert the image to Base64 for display
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Predicted: {result}")
    ax.axis('off')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return render_template('result.html', result=result, image_data=img_base64, video_path=None)

def process_video(filepath, filename):
    # Process the video
    cap = cv2.VideoCapture(filepath)
    crash_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized = frame_resized / 255.0
        frame_resized = np.expand_dims(frame_resized, axis=0)

        # Make a prediction
        prediction = model.predict(frame_resized)
        if (prediction > 0.5).astype(int)[0][0] == 1:
            crash_detected = True
            break

    cap.release()
    result = 'Crash' if crash_detected else 'No Crash'

    # Store crash data
    crash_data.append(result)

    return render_template('result.html', result=result, image_data=None, video_path=url_for('static', filename=f'uploads/{filename}'))

# Dashboard page
@app.route('/dashboard')
def dashboard():
    # Convert crash data to a format suitable for charting
    crash_count = sum(1 for crash in crash_data if crash == 'Crash')
    no_crash_count = len(crash_data) - crash_count
    data = {
        'labels': ['Crash', 'No Crash'],
        'counts': [crash_count, no_crash_count]
    }
    return render_template('dashboard.html', data=data)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
