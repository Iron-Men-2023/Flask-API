import base64
import json

import cv2
import requests
import time


class FacialRecognitionAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def recognize_face(self, path, user_id, device_sent_from="web"):
        url = f"{self.base_url}/api/facial_recognition"
        with open(path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        data = {"image": encoded_image, "user_id": user_id, "num_of_faces": 1, "device_sent_from": device_sent_from}
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

    def train_data_post(self):
        url = f"{self.base_url}/api/train_data"
        data = {"user_id": "Hi"}
        try:
            response = requests.post(url, json=data)

            if response.status_code == 200:
                data = response.json()
                print(data)
                return data

            else:
                print(f"Request failed with status code {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

    def train_one_image(self, path='ImagesTest/Kelly.jpeg', user_id='LfqBYBcq1BhHUvmE7803PhCFxeI2',
                        device_sent_from="web", name="Unknown"):
        url = f"{self.base_url}/api/facial_recognition/check"
        with open(path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        data = {"image": encoded_image, "user_id": user_id, "num_of_faces": 1, "device_sent_from": device_sent_from,
                'name': name}
        try:
            response = requests.post(url, json=data)

            if response.status_code == 200:
                data = response.json()
                print(data)
                return data

            else:
                print(f"Request failed with status code {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None


# http://192.168.0.233:8000
# https://flask-api-omnilense.herokuapp.com
api = FacialRecognitionAPI("http://192.168.0.119:8000")
user_id = "LfqBYBcq1BhHUvmE7803PhCFxeI2"
path = "imagesTest/Kelly.jpeg"

start = time.time()
# result = api.recognize_face(path, user_id, "web")
# result = api.train_data_post()
result = api.train_one_image(path, user_id, "web", "Kelly")
print(time.time() - start)
print(result)
