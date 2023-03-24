import json

import requests
import time


class FacialRecognitionAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def recognize_face(self, path, user_id):
        url = f"{self.base_url}/api/facial_recognition"
        data = {"path": path}
        data = json.dumps(data)

        try:
            response = requests.post(url, json=data)

            if response.status_code == 200:
                data = response.json()
                print(data)
                if data.get("message") == "No face found" or not data.get("predicted_person") or data.get(
                        "message") == "Error":
                    print(data.get("message"))
                    return "Unknown"

                predicted_person = data.get("predicted_person")[0]
                if predicted_person == "Unknown":
                    return predicted_person

                name = " ".join(predicted_person.split("_"))
                return name

            else:
                print(f"Request failed with status code {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None


# https://flask-api-omnilense.herokuapp.com
# http://172.17.117.8:8000
api = FacialRecognitionAPI("http://172.17.117.8:8000")
# Example of a user ID
user_id = "LfqBYBcq1BhHUvmE7803PhCFxeI2"
path = "images/ml_images/{}.jpg".format(user_id)
start = time.time()
result = api.recognize_face(path, user_id)
print(time.time() - start)
print(result)
