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
            img_encoding_list = face_recognition.face_encodings(rgb_img, num_jitters=100)
            if len(img_encoding_list) == 0:
                print(f"No face found in {img_path}. Enhancing image.")
                enhanced_img = self.enhance_image(img_resize)
                rgb_enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                img_encoding_list = face_recognition.face_encodings(rgb_enhanced_img, num_jitters=50)

            if len(img_encoding_list) > 0:
                img_encoding = img_encoding_list[0]

                # Store file name and file encoding
                self.encodings.append(img_encoding)
                self.names.append(filename)
                # Save the encoding to a file
                # np.save("encodings/{}.npy".format(filename), img_encoding)
            else:
                print(f"No face found in {img_path} even after enhancement.")
        np.save("encodings/{}.npy".format('trainedEncodings'),  self.encodings)
        # save the names to a file
        np.save("encodings/{}.npy".format('trainedNames'), self.names)
        print("Encoding images loaded")

    def load_single_img(self, path):
        # Loading images from folder
        img = cv2.imread(path)
        img_resize = cv2.resize(img, (0, 0), fx=self.resizedFrame, fy=self.resizedFrame)
        rgb_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        # Get the filename only from the initial file path.
        basename = os.path.basename(path)
        (filename, ext) = os.path.splitext(basename)

        # Get encoding
        img_encoding_list = face_recognition.face_encodings(rgb_img, num_jitters=100)
        if len(img_encoding_list) == 0:
            print(f"No face found in {path}. Enhancing image.")
            enhanced_img = self.enhance_image(img_resize)
            rgb_enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            img_encoding_list = face_recognition.face_encodings(rgb_enhanced_img, num_jitters=50)

        if len(img_encoding_list) > 0:
            img_encoding = img_encoding_list[0]

            # Store file name and file encoding
            self.encodings.append(img_encoding)
            self.names.append(filename)
            return True
        else:
            print(f"No face found in {path} even after enhancement.")

        print("Encoding images loaded")
        return False

    def detect_known_faces(self, frame):
        print("Detecting faces")
        if frame is None:
            print("Frame is None")
            return None, None
        print("Frame size: {}".format(frame.shape))
        small_frame = cv2.resize(frame, (0, 0), fx=self.resizedFrame, fy=self.resizedFrame)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=4, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=60)

        face_names = []
        if len(face_encodings) > 0:
            for face_encoding in face_encodings:
                # load encodings from file
                encodings = np.load("encodings/trainedEncodings.npy", allow_pickle=True)
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                # load the names from file
                names = np.load("encodings/trainedNames.npy", allow_pickle=True)
                if matches[best_match_index]:
                    name = names[best_match_index]
                face_names.append(name)
                print("Face found: {}".format(name))

            print("Face locations: {}".format(face_locations))
            # Convert to a dictionary and label which ones are top, right, bottom, left
            face_locations = np.array(face_locations)
            face_locations = (face_locations / self.resizedFrame).astype(int)

            face_data = [{"name": name, "location": {"top": loc[0], "right": loc[1], "bottom": loc[2], "left": loc[3]}}
                         for name, loc in zip(face_names, face_locations)]
            return face_data
        else:
            print("No face found. Enhancing image.")
            # enhanced_small_frame = self.enhance_image(small_frame)
            # rgb_enhanced_small_frame = cv2.cvtColor(enhanced_small_frame, cv2.COLOR_BGR2RGB)
            # face_locations = face_recognition.face_locations(rgb_enhanced_small_frame)
            # face_encodings = face_recognition.face_encodings(rgb_enhanced_small_frame, face_locations)
            #
            # if len(face_encodings) > 0:
            #     for face_encoding in face_encodings:
            #         # See if the face is a match for the known face(s)
            #         matches = face_recognition.compare_faces(self.encodings, face_encoding)
            #         name = "Unknown"
            #
            #         # Or instead, use the known face with the smallest distance to the new face
            #         face_distances = face_recognition.face_distance(self.encodings, face_encoding)
            #         best_match_index = np.argmin(face_distances)
            #         if matches[best_match_index]:
            #             name = self.names[best_match_index]
            #         face_names.append(name)
            #         print("Face found: {}".format(name))
            #
            #     face_locations = np.array(face_locations)
            #     face_locations = (face_locations / self.resizedFrame).astype(int)
            #
            #     face_data = [
            #         {"name": name, "location": {"top": loc[0], "right": loc[1], "bottom": loc[2], "left": loc[3]}}
            #         for name, loc in zip(face_names, face_locations)]
            #     return face_data
            # else:
            #     print("No face found even after enhancement.")
            return None
