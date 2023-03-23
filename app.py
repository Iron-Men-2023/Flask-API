import json

from flask import Flask, request, jsonify, render_template
from downloadImgAndRec import FirebaseImageRecognizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
recognizer = FirebaseImageRecognizer("omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json",
                                     "omnilens-d5745.appspot.com")


@app.route('/api/facial_recognition', methods=['POST'])
def facial_recognition():
    # Get image data from request
    print(request.data)
    requestForm = request.data.decode('utf-8')  # Decode using the utf-8 encoding
    requestForm = json.loads(requestForm)  # Convert to JSON
    path = requestForm['path']
    user_id = requestForm['user_id']
    print("Path: ", path)
    print("User ID: ", user_id)
    if user_id is None:
        face_names, recents = recognizer.process_image(path, None)
    else:
        face_names, recents = recognizer.process_image(path, user_id)

    if face_names is None:
        return jsonify({'message': 'No face found'})
    else:
        try:
            # faceLoc_dict = {}
            # for i in range(len(faceLoc)):
            #     faceLoc_dict[i] = faceLoc[i]
            # # Convert the NumPy arrays to nested lists using tolist()
            # data_json = json.dumps(faceLoc[0].tolist())
            # Convert the list to a JSON string
            if recents is None:
                recents_json = "Error getting recents or not entered"
            else:
                recents_json = json.dumps(recents)

            # Print the JSON string
            print(face_names)
            jsonConv = jsonify({'predicted_person': face_names, 'recents': recents_json, 'message': 'Face found'})
            print("Json: ", jsonConv)
            return jsonConv
        except Exception as e:
            print("Error: ", e)
            return jsonify({'message': 'Error'})


@app.route('/api/facial_recognition', methods=['OPTIONS'])
def handle_options():
    response = app.make_default_options_response()

    # Set the Access-Control-Allow-Headers header
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

    return response


@app.route('/api/facial_recognition', methods=['GET'])
def get():
    return jsonify({'message': 'Hello World!'})


# A new default route
@app.route('/')
def intro():
    return render_template('intro.html')


@app.route('/documentation')
def documentation():
    # Replace with the code to render your documentation page
    return render_template('doc.html')


@app.route('/help')
def help():
    # Replace with the code to render your help page
    return render_template('help.html')


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(port=8000, debug=True, threaded=True, host='0.0.0.0')
