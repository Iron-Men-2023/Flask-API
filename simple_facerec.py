import face_recognition
import cv2
import os
import glob
import numpy as np


class RecognitionHelper:
    def __init__(self):
        self.names = []
        self.encodings = []
        # Resize frame for a faster speed
        self.resizedFrame = 0.25

    def load_images(self, images_path):
        # Loading images from folder
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, (0, 0), fx=self.resizedFrame, fy=self.resizedFrame)
            rgb_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Get encoding
            img_encoding_list = face_recognition.face_encodings(rgb_img)
            if len(img_encoding_list) > 0:
                img_encoding = img_encoding_list[0]

                # Store file name and file encoding
                self.encodings.append(img_encoding)
                self.names.append(filename)
            else:
                print(f"No face found in {img_path}")

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        print("Detecting faces")
        if frame is None:
            print("Frame is None")
            return None, None
        print("Frame size: {}".format(frame.shape))
        small_frame = cv2.resize(frame, (0, 0), fx=self.resizedFrame, fy=self.resizedFrame)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        if len(face_encodings) > 0:
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.names[best_match_index]
                face_names.append(name)
                print("Face found: {}".format(name))

            # Convert to numpy array to adjust coordinates with frame resizing quickly
            face_locations = np.array(face_locations)
            face_locations = face_locations / self.resizedFrame
            return face_locations.astype(int), face_names
        else:
            print("No face found")
            return None, None
