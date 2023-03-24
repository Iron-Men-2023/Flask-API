import os
import cv2
import numpy as np
from google.cloud import storage
from firebase_admin import credentials, initialize_app, firestore


class FirebaseImageDownloader:
    def __init__(self, firebase_json_key, firebase_bucket_name, download_folder='images/'):
        self.firebase_json_key = firebase_json_key
        self.firebase_bucket_name = firebase_bucket_name
        self.download_folder = download_folder

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.firebase_json_key

        cred = credentials.Certificate(self.firebase_json_key)
        initialize_app(cred, {'storageBucket': self.firebase_bucket_name})
        self.client = storage.Client()
        self.db = firestore.client()
        self.bucket = self.client.get_bucket(self.firebase_bucket_name)

    def download_blobs_in_folder(self, folder):
        blobs = self.bucket.list_blobs(prefix=folder)
        for blob in blobs:
            if not blob.name.endswith('/'):
                print("Blob name: ", blob.name)
                start_index = blob.name.index("/")
                end_index = blob.name.index(".")
                result = blob.name[start_index:end_index]
                name = result.split("/")[2]
                print('name', name)
                user_ref = self.db.collection('users').document(name).get()
                if not user_ref.exists:
                    continue
                try:
                    user = user_ref.to_dict()['name']
                    print('user', user)
                    user = user.replace(" ", "_")

                    image_bytes = blob.download_as_bytes()

                    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
                    image = cv2.imdecode(image_array, -1)

                    save_path = os.path.join(self.download_folder, f"{user}.jpg")
                    print(save_path)
                    if not os.path.exists(save_path):
                        cv2.imwrite(save_path, image)
                        print("Saved image to: ", save_path)
                    else:
                        print("File already exists")
                except KeyError:
                    continue


if __name__ == "__main__":
    firebase_json_key = 'omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json'
    firebase_bucket_name = 'omnilens-d5745.appspot.com'
    downloader = FirebaseImageDownloader(firebase_json_key, firebase_bucket_name)
    downloader.download_blobs_in_folder('images/Avatar')
