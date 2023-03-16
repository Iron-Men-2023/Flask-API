from datetime import datetime
import cv2
from simple_facerec import RecognitionHelper
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage, firestore

import os
import time


def upscale_image(image, method):
    # Upscale the image
    result = cv2.resize(image, None, fx=4, fy=4, interpolation=getattr(cv2, method))
    return result


class FirebaseImageRecognizer:
    def __init__(self, credential_path, storage_bucket):
        self.cred = credentials.Certificate(credential_path)
        self.app = firebase_admin.initialize_app(self.cred, {'storageBucket': storage_bucket}, name='storage')
        self.bucket = storage.bucket(app=self.app)
        self.db = firestore.client(app=self.app)
        self.sfr = RecognitionHelper()
        self.sfr.load_images("images")

    def get_all_images(self):
        prefix = "images/Avatar"
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            # Process and save images here
            pass
        return blobs

    def save_image(self, image, save_path):
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, image)
        else:
            print("File already exists")

    def try_recognition(self, image, methods):
        for method in methods:
            upscaled_image = upscale_image(image, method)
            face_locations, face_names = self.sfr.detect_known_faces(upscaled_image)
            if face_names:
                print(f"Found face using {method} interpolation")
                print(face_names)
                return face_names, face_locations

        print("No face found")
        return "No face found", None

    def process_image(self, image_path):
        blob = self.bucket.blob(image_path)
        expiration = int(datetime.now().timestamp() + 600)
        print(blob.generate_signed_url(expiration))

        image_bytes = blob.download_as_bytes()
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(image_array, -1)

        save_path = os.path.join("imagesTest", "test.png")
        self.save_image(image, save_path)

        methods = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"]
        start_time = time.time()
        elapsed_time = 0
        while elapsed_time < 5:
            face_names, face_locations = self.try_recognition(image, methods)
            if face_names != "No face found":
                return face_names, face_locations

            elapsed_time = time.time() - start_time

        return "No face found", None
