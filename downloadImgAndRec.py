import time
from datetime import datetime
import cv2
from simple_facerec import RecognitionHelper
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os

def upscale_image(image, method):
    result = cv2.resize(image, None, fx=4, fy=4, interpolation=getattr(cv2, method))
    return result


class FirebaseImageRecognizer:
    def __init__(self, credential_path, storage_bucket):
        self.cred = credentials.Certificate(credential_path)
        self.app = firebase_admin.initialize_app(self.cred, {'storageBucket': storage_bucket}, name='storage')
        self.bucket = storage.bucket(app=self.app)
        self.db = firestore.client(app=self.app)
        self.sfr = RecognitionHelper()
        # self.sfr.load_images("images")

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)

    def try_recognition(self, image, methods):
        for method in methods:
            face_data = self.sfr.detect_known_faces(image)
            if face_data:
                print(f"Found face using {method} interpolation")
                print(face_data)
                return face_data

        print("No face found")
        return None

    def process_image(self, image_path, user_id, num_of_faces):
        # blob = self.bucket.blob(image_path)
        # expiration = int(datetime.now().timestamp() + 600)
        # print(blob.generate_signed_url(expiration))
        #
        # image_bytes = blob.download_as_bytes()
        # image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        # image = cv2.imdecode(image_array, -1)
        #
        # save_path = os.path.join("imagesTest", "test.png")
        # self.save_image(image, save_path)
        image = cv2.imread(image_path)
        face_data = self.sfr.detect_known_faces(image, num_of_faces)
        if face_data:
            print("Face found in image: ", face_data)
            # if user_id is not None:
            #     try:
            #         recents = self.add_to_recents(user_id, [face["name"] for face in face_data])
            #     except Exception as e:
            #         print(e)
            #         recents = None
            # else:
            #     recents = None
            return face_data, None
        # methods = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"]
        # start_time = time.time()
        # elapsed_time = 0
        # while elapsed_time < 6:
        #     face_data = self.try_recognition(image, methods)
        #     if face_data:
        #         print("Face found in image: ", face_data)
        #         if user_id is not None:
        #             recents = self.add_to_recents(user_id, [face["name"] for face in face_data])
        #         else:
        #             recents = None
        #         return face_data, recents
        #
        #     elapsed_time = time.time() - start_time

        return None, None

    def add_to_recents(self, user_id, face_names):
        users_ref = self.db.collection('users')
        users = users_ref.stream()
        for user in users:
            if user.get("name") in face_names:
                recent_ref = self.db.collection('users').document(user_id)
                recent_ref.update({
                    u'recents': firestore.ArrayUnion([user.id])
                })
                return recent_ref.get().get("recents")
        return None

    def download_image(self, image_path, save_path):
        # Keep trying to download image until it works
        elapsed_time = 0
        start_time = time.time()
        while True or elapsed_time < 10:
            try:
                blob = self.bucket.blob(image_path)
                expiration = int(datetime.now().timestamp() + 600)
                print(blob.generate_signed_url(expiration))

                image_bytes = blob.download_as_bytes()
                image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
                image = cv2.imdecode(image_array, -1)

                self.save_image(image, save_path)
                return image
            except Exception as e:
                time.sleep(1)
                print(e)
                elapsed_time = time.time() - start_time
