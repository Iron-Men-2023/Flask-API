import os

import cv2
import numpy as np
from google.cloud import storage
from firebase_admin import credentials, initialize_app, firestore

# Replace with the path to your Firebase JSON key
FIREBASE_JSON_KEY = '../omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json'

# Replace with your Firebase bucket name
FIREBASE_BUCKET_NAME = 'omnilens-d5745.appspot.com'
# Replace with your desired local download folder
DOWNLOAD_FOLDER = 'images/'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = FIREBASE_JSON_KEY

cred = credentials.Certificate(FIREBASE_JSON_KEY)
initialize_app(cred, {'storageBucket': FIREBASE_BUCKET_NAME})
client = storage.Client()
db = firestore.client()
bucket = client.get_bucket(FIREBASE_BUCKET_NAME)


def download_blobs_in_folder(folder):
    blobs = bucket.list_blobs(prefix=folder)
    for blob in blobs:
        if not blob.name.endswith('/'):
            print("Blob name: ", blob.name)
            start_index = blob.name.index("/")
            end_index = blob.name.index(".")
            result = blob.name[start_index:end_index]
            name = result.split("/")[2]
            print('name', name)
            # Get the user associated with the image name from the Firebase database
            user_ref = db.collection('users').document(name).get()
            # If the user doesn't exist, skip the image
            if not user_ref.exists:
                continue
            try:
                user = user_ref.to_dict()['name']
                print('user', user)
                user = user.replace(" ", "_")

                # Download the image as a bytes object
                image_bytes = blob.download_as_bytes()

                image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
                image = cv2.imdecode(image_array, -1)

                # Save the original image with the user's name as the path
                save_path = os.path.join("images", f"{user}.jpg")
                print(save_path)
                if not os.path.exists(save_path):
                    cv2.imwrite(save_path, image)
                    print("Saved image to: ", save_path)
                else:
                    print("File already exists")
            except KeyError:
                continue


if __name__ == "__main__":
    download_blobs_in_folder('images/Avatar')
