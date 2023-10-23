from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import cv2
import pickle
import os
import base64
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import csv
import time
from datetime import datetime
from retinaface import RetinaFace
import matplotlib.pyplot as plt
from deepface import DeepFace

app = Flask(__name__)

# Folder to save captured images
DATA_FOLDER = "/Users/apple/Desktop/Face_Detection_FlaskApp/data"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_user', methods=['GET'])
def new_user():
    return render_template('new_user.html')


@app.route('/capture', methods=['POST'])
def capture():
    # Get the candidate's name from the form
    name = request.form['name']

    # Check if DATA_FOLDER exists, if not create it
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    captured_faces = []

    # Save the captured images and immediately read and process them
    for key in request.form:
        if key.startswith('image'):
            image_data = base64.b64decode(request.form[key].split(',')[1])
            image_filename = f"{name}_{key}.png"
            image_path = os.path.join(DATA_FOLDER, image_filename)
            with open(image_path, 'wb') as image_file:
                image_file.write(image_data)
            # Read the image and convert to grayscale if it's not already
            face = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            captured_faces.append(face)

    # Check if faces.pkl and names.pkl exist, if not create them with empty lists
    if not os.path.exists(os.path.join(DATA_FOLDER, 'faces.pkl')):
        with open(os.path.join(DATA_FOLDER, 'faces.pkl'), 'wb') as file:
            pickle.dump([], file)
    if not os.path.exists(os.path.join(DATA_FOLDER, 'names.pkl')):
        with open(os.path.join(DATA_FOLDER, 'names.pkl'), 'wb') as file:
            pickle.dump([], file)

    # Load existing faces and names
    with open(os.path.join(DATA_FOLDER, 'faces.pkl'), 'rb') as file:
        existing_faces = pickle.load(file)
    with open(os.path.join(DATA_FOLDER, 'names.pkl'), 'rb') as file:
        existing_names = pickle.load(file)

    # Update faces and names lists
    existing_faces.extend(captured_faces)
    existing_names.extend([name] * len(captured_faces))

    # Save the updated data
    with open(os.path.join(DATA_FOLDER, 'names.pkl'), 'wb') as w:
        pickle.dump(existing_names, w)
    with open(os.path.join(DATA_FOLDER, 'faces.pkl'), 'wb') as f:
        pickle.dump(existing_faces, f)

    return jsonify({"message": "Photos captured and saved!"})


@app.route('/images', methods=['GET'])
def show_images():
    # Retrieve image filenames from DATA_FOLDER
    image_filenames = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.png')]
    return render_template('show_images.html', images=image_filenames)

@app.route('/data/<filename>', methods=['GET'])
def send_image(filename):
    return send_from_directory(DATA_FOLDER, filename)

def convert_int64_to_int(obj):
    """Recursively convert int64 values in a data structure to int."""
    if isinstance(obj, dict):
        return {k: convert_int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

@app.route('/ml_prediction', methods=['GET', 'POST'])
def ml_prediction():
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Convert each image in FACES to grayscale only if it's not already grayscale
    FACES = [face if len(face.shape) == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in FACES]
    
    # Convert FACES to numpy array and reshape
    FACES = np.array(FACES).reshape(len(FACES), -1)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    if request.method == 'POST':
        # Decode the received image
        image_data = base64.b64decode(request.form['image'].split(',')[1])
        image_np = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Save the received image with a name based on the current time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"Prediction_{current_time}.png"
        cv2.imwrite(img_path, frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        predictions = []
        for (x, y, w, h) in faces:
            crop_img = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (640, 480)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            predictions.append({
                "name": output[0],
                "box": [int(x), int(y), int(w), int(h)]
            })

        print(len(predictions))
        print(predictions)

        # If there are more than one predictions, keep only the first one
        if len(predictions) > 1:
            predictions = [predictions[0]]

        resp = RetinaFace.detect_faces(img_path)

        print(resp)

        retinaface_predictions = []
        for key in resp.keys():
            identity = resp[key]
            facial_area = identity["facial_area"]
            retinaface_predictions.append({
                # "name": "Face Detected",
                "box": [facial_area[2], facial_area[3], facial_area[0]-facial_area[2], facial_area[1]-facial_area[3]]
            })

        folder_path = DATA_FOLDER
        best_match_distance = float('inf')
        best_match_img_path = None
        best_match_file = None

        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                current_img_path = os.path.join(folder_path, file)
                print(current_img_path)
                if current_img_path != img_path:
                    obj = DeepFace.verify(img1_path=img_path, img2_path=current_img_path, model_name="ArcFace", detector_backend="retinaface")
                    if 'distance' in obj and obj['distance'] < best_match_distance:
                        best_match_distance = obj['distance']
                        best_match_img_path = current_img_path
                        best_match_file = file

        # predictions.append({
        #     "name": f"Best Match: {os.path.basename(best_match_img_path)}",
        #     "box": [0, 0, 0, 0]
        # })

        print(len(retinaface_predictions))
        print(retinaface_predictions)

        if best_match_file and retinaface_predictions:
            retinaface_predictions[0]["name"] = best_match_file


        combined_predictions = {
            "knn": predictions,
            "retinaface": retinaface_predictions
        }

        print(combined_predictions)

        return jsonify(convert_int64_to_int(combined_predictions))

    return render_template('ml_predictions.html')


if __name__ == '__main__':
    app.run(debug=False)
