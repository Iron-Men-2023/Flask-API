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

    def enhance_image(self, img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        enhanced_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return enhanced_img

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
            if len(img_encoding_list) == 0:
                print(f"No face found in {img_path}. Enhancing image.")
                enhanced_img = self.enhance_image(img_resize)
                rgb_enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                img_encoding_list = face_recognition.face_encodings(rgb_enhanced_img)

            if len(img_encoding_list) > 0:
                img_encoding = img_encoding_list[0]

                # Store file name and file encoding
                self.encodings.append(img_encoding)
                self.names.append(filename)
            else:
                print(f"No face found in {img_path} even after enhancement.")

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        print("Detecting faces")
        if frame is None:
            print("Frame is None")
            return None, None
        print("Frame size: {}".format(frame.shape))
        small_frame = cv2.resize(frame, (0, 0), fx=self.resizedFrame, fy=self.resizedFrame)
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
            print("No face found. Enhancing image.")
            enhanced_small_frame = self.enhance_image(small_frame)
            rgb_enhanced_small_frame = cv2.cvtColor(enhanced_small_frame, cv2.COLOR_BGR2RGB)
            enhanced_face_locations = face_recognition.face_locations(rgb_enhanced_small_frame)
            enhanced_face_encodings = face_recognition.face_encodings(rgb_enhanced_small_frame, enhanced_face_locations)

            if len(enhanced_face_encodings) > 0:
                for face_encoding in enhanced_face_encodings:
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
                enhanced_face_locations = np.array(enhanced_face_locations)
                enhanced_face_locations = enhanced_face_locations / self.resizedFrame
                return enhanced_face_locations.astype(int), face_names
            else:
                print("No face found even after enhancement.")
                return None, None
