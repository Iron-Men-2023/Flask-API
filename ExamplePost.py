import requests

class FacialRecognitionAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def recognize_face(self, path):
        url = f"{self.base_url}/api/facial-recognition"
        data = {"path": path, "user_id": "LfqBYBcq1BhHUvmE7803PhCFxeI2"}

        try:
            response = requests.post(url, data=data)

            if response.status_code == 200:
                data = response.json()
                if data.get("message") == "No face found" or not data.get("predicted_person") or data.get("message") == "Error":
                    print(data.get("message"))
                    return None

                predicted_person = data.get("predicted_person")[0]
                if predicted_person == "Unknown":
                    return predicted_person

                name = " ".join(predicted_person.split("_")[1:])
                return name

            else:
                print(f"Request failed with status code {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None


api = FacialRecognitionAPI("http://192.168.0.119:8000")
# Example of a user ID
user_id = "LfqBYBcq1BhHUvmE7803PhCFxeI2"
path = "images/ml_images/{}.jpg".format(user_id)
result = api.recognize_face(path)
print(result)