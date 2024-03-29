import json
import math
import time
import os

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from downloadImgAndRec import FirebaseImageRecognizer
from download_images_from_firebase import FirebaseImageDownloader
from simple_facerec import RecognitionHelper
from flask_cors import CORS
import base64
import io
from PIL import Image
import ujson

app = Flask(__name__)
CORS(app, supports_credentials=True)
recognizer = FirebaseImageRecognizer("omnilens-d5745-firebase-adminsdk-rorof-6a932e9332.json",
                                     "omnilens-d5745.appspot.com")
recognizerHelper = RecognitionHelper()


@app.route('/api/facial_recognition', methods=['POST'])
def facial_recognition():
    start_time = time.time()
    try:
        form = request.form.to_dict()
        user_id = form['user_id']
        device_sent_from = form['device_sent_from']
        print("Device sent from: ", device_sent_from)
        image_base64 = form['image']
        num_of_faces = int(form['num_of_faces'])
    except:
        try:
            start_time1 = time.time()
            # jsonData = request.data.decode('utf-8')
            jsonData = ujson.loads(request.data)
            image_base64 = jsonData['image']
            device_sent_from = jsonData['device_sent_from']
            user_id = jsonData['user_id']
            num_of_faces = int(jsonData['num_of_faces'])
            print("Time taken to decode: ", time.time() - start_time1)

        except:
            raise Exception("Error getting data from request")
    if device_sent_from == "test":
        image_path = "imagesTest/Kelly.jpeg"
    else:
        print("base64 image: ", image_base64[:100])
        image = base64_to_image(image_base64)
        image_path = save_image(image, user_id, device_sent_from)
    print("Time taken to get image: ", time.time() - start_time)
    start1 = time.time()
    face_data, recents = recognizer.process_image(image_path, user_id, num_of_faces)
    print("Time taken to process image: ", time.time() - start1)
    print("Face data from class: ", face_data)

    if face_data is None:
        return jsonify({'message': 'No face found'})
    else:
        try:
            face_names = [face["name"] for face in face_data]
            face_loc = [face["location"] for face in face_data]
            confidence = [face["confidence"] for face in face_data]

            face_loc = json.dumps(face_loc, default=lambda x: int(x) if isinstance(x, np.integer) else x)
            if recents is None:
                recents_json = "Error getting recents or not entered"
            else:
                recents_json = json.dumps(recents)

            response_data = {'predicted_person': face_names, 'recents': recents_json, "face_loc": face_loc,
                             "confidence": confidence,
                             'message': 'Face found'}
            jsonConv = jsonify(response_data)

            return jsonConv
        except Exception as e:
            print("Error: ", e)
            return jsonify({'message': 'Error'})


def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img


def save_image(image, user_id, device_sent_from):
    # if device_sent_from == "app":
    # image = image.rotate(180, expand=True)
    filename = f'imagesTest/{user_id}.jpg'
    if device_sent_from == "app":
        print("Changing image color")
        image = image.convert('RGB')  # Convert the image to RGB mode
    # make image bigger
    image = cv2.resize(np.array(image), (0, 0), fx=3, fy=3)
    cv2.imwrite(filename, image)
    return filename


@app.route('/api/facial_recognition', methods=['OPTIONS'])
def handle_options():
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


def train():
    recognizerHelper.train_on_images("static/images")


@app.route('/api/train_data', methods=['POST'])
def train_data():
    firebase_json_key = 'omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json'
    firebase_bucket_name = 'omnilens-d5745.appspot.com'
    downloader = FirebaseImageDownloader(firebase_json_key, firebase_bucket_name)
    downloader.download_blobs_in_folder('images/Avatar')
    train()
    return jsonify({'message': 'Training done'})


@app.route('/test_faces')
def test_faces():
    folder_path = 'static/images'
    image_paths = []
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        # Go through each image in the folder and get the full path
        for image_filename in os.listdir(folder_path_full):
            image_path = os.path.join(folder_path_full, image_filename)
            image_paths.append(image_path)
            break
    predictions = []
    confidence_list = []
    for image_path in image_paths:
        print("Image path: ", image_path)
        # Get the prediction for the image
        face_data, _ = recognizer.process_image(image_path, None, 1)
        if face_data is not None:
            predictions.append(face_data[0]['name'])
            confidence = round(face_data[0]['confidence'], 2)
            confidence_list.append(confidence)
        else:
            predictions.append('Unknown')
            confidence_list.append(0.00)

    image_prediction_pairs = zip(image_paths, predictions, confidence_list)
    return render_template('test_faces.html', image_prediction_pairs=image_prediction_pairs)


@app.route('/api/facial_recognition/feedback', methods=['POST'])
def submit_feedback():
    # print(request.data)
    feedback_data = request.get_json()
    image_path = feedback_data['image_path']
    answer = feedback_data['answer']
    prediction = feedback_data['prediction']
    print("Image path: ", image_path)
    print("Feedback data: ", feedback_data)
    update_model = recognizerHelper.update_model(feedback_data)
    if update_model:
        return jsonify({'message': f'Feedback received: {answer} for image {image_path} with prediction {prediction}',
                        'delete': False})
    else:
        # Delete the image from the database
        recognizerHelper.delete_image(image_path)
        return jsonify({'message': 'Image did not have a recognised face. Image deleted from database', 'delete': True})


@app.route('/api/facial_recognition', methods=['GET'])
def get():
    # Reroute to intro page
    return render_template('intro.html')


@app.route('/api/facial_recognition/check', methods=['POST'])
def check():
    start_time = time.time()
    try:
        form = request.form.to_dict()
        user_id = form['user_id']
        device_sent_from = form['device_sent_from']
        print("Device sent from: ", device_sent_from)
        image_base64 = form['image']
        name = form['name']
    except:
        try:
            start_time1 = time.time()
            # jsonData = request.data.decode('utf-8')
            jsonData = ujson.loads(request.data)
            image_base64 = jsonData['image']
            device_sent_from = jsonData['device_sent_from']
            user_id = jsonData['user_id']
            name = jsonData['name']
            print("Time taken to decode: ", time.time() - start_time1)

        except:
            raise Exception("Error getting data from request")
    print("Device Sent from: ", device_sent_from)
    image = base64_to_image(image_base64)
    image_path = save_image(image, user_id, device_sent_from)
    print("Image path: ", image_path)
    # Check if there is a face in the image
    check_face = recognizerHelper.load_single_img(image_path)
    print("Time taken to get image: ", time.time() - start_time)
    if check_face is None:
        print("No face found")
        return jsonify({'message': 'No face found'})
    else:
        print("Face found")
        recognizerHelper.train_one_image_add_to_encodings(image_path, name)
        return jsonify({'message': 'Face found'})


@app.route('/')
def intro():
    return render_template('intro.html')


@app.route('/documentation')
def documentation():
    return render_template('doc.html')


@app.route('/help')
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(port=8000, debug=True, threaded=True, host='0.0.0.0')
