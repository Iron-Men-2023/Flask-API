import json
import time

import numpy as np
from flask import Flask, request, jsonify, render_template
from downloadImgAndRec import FirebaseImageRecognizer
from simple_facerec import RecognitionHelper
from flask_cors import CORS
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app, supports_credentials=True)
recognizer = FirebaseImageRecognizer("omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json",
                                     "omnilens-d5745.appspot.com")


@app.route('/api/facial_recognition', methods=['POST'])
def facial_recognition():
    print("Request: ", request.data)
    try:
        jsonData = request.data.decode('utf-8')
        jsonData = json.loads(jsonData)
        image_base64 = jsonData['image']

    except:
        jsonData = request.get_json()
        jsonData = json.loads(jsonData)
        image_base64 = jsonData['image']

    user_id = jsonData['user_id']
    image = base64_to_image(image_base64)
    image_path = save_image(image, user_id)
    print(type(jsonData))
    try:
        user_id = jsonData['user_id']
    except KeyError:
        user_id = None
    try:
        num_of_faces = jsonData['num_of_faces']
    except KeyError:
        num_of_faces = 1
    start = time.time()
    face_data, recents = recognizer.process_image(image_path, user_id, num_of_faces)
    print("Time taken to process image: ", time.time() - start)
    print("Face data from class: ", face_data)

    if face_data is None:
        return jsonify({'message': 'No face found'})
    else:
        try:
            face_names = [face["name"] for face in face_data]
            face_loc = [face["location"] for face in face_data]

            face_loc = json.dumps(face_loc, default=lambda x: int(x) if isinstance(x, np.integer) else x)
            if recents is None:
                recents_json = "Error getting recents or not entered"
            else:
                recents_json = json.dumps(recents)

            response_data = {'predicted_person': face_names, 'recents': recents_json, "face_loc": face_loc,
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


def save_image(image, user_id):
    # rotate image 90 degrees
    image = image.rotate(270, expand=True)
    filename = f'imagesTest/{user_id}.jpg'
    image.save(filename, 'JPEG')
    return filename


@app.route('/api/facial_recognition', methods=['OPTIONS'])
def handle_options():
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/api/facial_recognition', methods=['GET'])
def get():
    # Reroute to intro page
    return render_template('intro.html')


@app.route('/api/facial_recognition/check', methods=['POST'])
def check():
    print(request.data)
    requestForm = request.data.decode('utf-8')
    requestForm = json.loads(requestForm)
    path = requestForm['path']
    user_id = requestForm['user_id']
    print("Path: ", path)
    download_to_path = "imagesTest/testProfile.{}.jpg".format(user_id)
    load_img = recognizer.download_image(path, download_to_path)
    # Check if there is a face in the image
    recognizerHelper = RecognitionHelper()
    check_face = recognizerHelper.load_single_img(download_to_path)
    if check_face is None:
        print("No face found")
        return jsonify({'message': 'No face found'})
    else:
        print("Face found")
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
