from datetime import datetime
import cv2
from simple_facerec import RecognitionHelper
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage, firestore

import os
import time


def upscale_image(image_path, method):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Upscale the image
    result = cv2.resize(image, None, fx=4, fy=4, interpolation=getattr(cv2, method))

    # Save the image
    save_path = f"imagesTest/{method}.png"
    cv2.imwrite(save_path, result)
    print(f"Saved image to: {save_path}")

    return save_path


class FirebaseImageRecognizer:
    def __init__(self, credential_path, storage_bucket):
        self.cred = credentials.Certificate(credential_path)
        self.app = firebase_admin.initialize_app(self.cred, {'storageBucket': storage_bucket}, name='storage')
        self.bucket = storage.bucket(app=self.app)
        self.db = firestore.client(app=self.app)
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel("EDSR_x4.pb")
        self.sr.setModel("edsr", 4)
        self.sfr = RecognitionHelper()
        self.sfr.load_images("images")

    def get_all_images(self):
        prefix = "images/Avatar"
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            print("Blob name: ", blob.name)
            start_index = blob.name.index("/")
            end_index = blob.name.index(".")
            result = blob.name[start_index:end_index]
            name = result.split("/")[2]
            print('name', name)
            # Get the user associated with the image name from the Firebase database
            user_ref = self.db.collection('users').document(name).get()
            if not user_ref.exists:
                continue
            user = user_ref.to_dict()['name']
            print('user', user)
            user = user.replace(" ", "_")

            # Download the image as a bytes object
            image_bytes = blob.download_as_bytes()

            image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            image = cv2.imdecode(image_array, -1)

            # Save the original image with the user's name as the path
            save_path = os.path.join("images", f"{user}.png")
            print(save_path)
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, image)
            else:
                print("File already exists")
        return blobs

    def recognize_faces(self, image_path):
        blob = self.bucket.blob(image_path)
        # Have time stamp be 10 minutes in the future
        expiration = int(datetime.now().timestamp() + 600)
        print(blob.generate_signed_url(expiration))

        # Download the image as a bytes object
        image_bytes = blob.download_as_bytes()

        # Convert the bytes object to a numpy array
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(image_array, -1)

        # Save the original image
        save_path = os.path.join("imagesTest", "test.png")
        cv2.imwrite(save_path, image)

        # Try all interpolation methods until a face is found or 5 seconds has passed
        start_time = time.time()
        elapsed_time = 0
        while elapsed_time < 5:
            for method in ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"]:
                try:
                    # Upscale the image using the current interpolation method
                    upscaled_image_path = upscale_image(save_path, method)

                    # Load the upscaled image from the file
                    upscaled_image = cv2.imread(upscaled_image_path)

                    # Initialize the face recognition model and detect known faces in the image
                    sfr = RecognitionHelper()
                    sfr.load_images("images")
                    face_locations, face_names = sfr.detect_known_faces(upscaled_image)

                    # Print the detected face names, or "No face found" if none were detected
                    print(face_names if face_names else f"No face found using {method} interpolation")

                    # Return the detected face names and locations if at least one face was found
                    if face_names:
                        print(f"Found face using {method} interpolation")
                        print(face_names)
                        return face_names, face_locations
                except ValueError as e:
                    # Continue to the next iteration if the image could not be loaded
                    print(e)
                    continue

            # Update the elapsed time
            elapsed_time = time.time() - start_time

        # Return "No face found" if no face was found after trying all methods for 5 seconds
        print("No face found")
        return "No face found", None
