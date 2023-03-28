import json
import time
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from downloadImgAndRec import FirebaseImageRecognizer
from simple_facerec import RecognitionHelper
from flask_cors import CORS
import base64
import io
from PIL import Image
import ujson

app = Flask(__name__)
CORS(app, supports_credentials=True)
recognizer = FirebaseImageRecognizer("omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json",
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
    image = base64_to_image(image_base64)
    image_path = save_image(image, user_id)
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


def save_image(image, user_id):
    # if device_sent_from == "app":
    # image = image.rotate(180, expand=True)
    filename = f'imagesTest/{user_id}.jpg'
    image.save(filename, 'JPEG')
    return filename


@app.route('/api/facial_recognition', methods=['OPTIONS'])
def handle_options():
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/test_faces')
def test_faces():
    image_folder = 'static/images'
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    predictions = []

    for image_path in image_paths:
        # Get the prediction for the image
        face_data, _ = recognizer.process_image(image_path, None, 1)
        if face_data is not None:
            predictions.append(face_data[0]['name'])
        else:
            predictions.append('Unknown')

    image_prediction_pairs = zip(image_paths, predictions)
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
    except:
        try:
            start_time1 = time.time()
            # jsonData = request.data.decode('utf-8')
            jsonData = ujson.loads(request.data)
            image_base64 = jsonData['image']
            device_sent_from = jsonData['device_sent_from']
            user_id = jsonData['user_id']
            print("Time taken to decode: ", time.time() - start_time1)

        except:
            raise Exception("Error getting data from request")
    image = base64_to_image(image_base64)
    image_path = save_image(image, user_id)
    print("Image path: ", image_path)
    # Check if there is a face in the image
    check_face = recognizerHelper.load_single_img(image_path)
    print("Time taken to get image: ", time.time() - start_time)
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
