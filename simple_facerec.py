import time
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
        self.resizedFrame = 0.20

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

            # Get location of face
            face_locations = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=3, model="cnn")
            # Get encoding
            img_encoding_list = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=200)
            # if len(img_encoding_list) == 0:
            #     print(f"No face found in {img_path}. Enhancing image.")
            #     enhanced_img = self.enhance_image(img_resize)
            #     rgb_enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            #     img_encoding_list = face_recognition.face_encodings(rgb_enhanced_img, num_jitters=150)

            if len(img_encoding_list) > 0:
                img_encoding = img_encoding_list[0]

                # Store file name and file encoding
                self.encodings.append(img_encoding)
                self.names.append(filename)
                print(f"Encoding {filename} done.")
                # Save the encoding to a file
                # np.save("encodings/{}.npy".format(filename), img_encoding)
            else:
                print(f"No face found in {img_path} even after enhancement.")
        np.save("encodings/{}.npy".format('trainedEncodings'), self.encodings)
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
        img_encoding_list = face_recognition.face_encodings(rgb_img, num_jitters=25)

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

    def detect_known_faces(self, frame, num_of_faces=1):
        print("Detecting faces")
        if frame is None:
            print("Frame is None")
            return None, None
        print("Frame size: {}".format(frame.shape))
        small_frame = cv2.resize(frame, (0, 0), fx=self.resizedFrame, fy=self.resizedFrame)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        if num_of_faces == 1:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        elif num_of_faces == 2:
            face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2,
                                                             model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=50)
        else:
            face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=3)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=60)

        if len(face_encodings) > 0:
            return self.process_encodings(face_locations, face_encodings)
        else:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=10)
            if len(face_encodings) > 0:
                return self.process_encodings(face_locations, face_encodings)
            else:
                print("No face found")
                return None

    def process_encodings(self, face_locations, face_encodings):
        face_names = []
        face_distances_list = []
        best_match_indexes_list = []
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
                face_distances_list.append(face_distances)
                best_match_indexes_list.append(best_match_index)

            print("Face locations: {}".format(face_locations))
            # Convert to a dictionary and label which ones are top, right, bottom, left
            face_locations = np.array(face_locations)
            face_locations = (face_locations / self.resizedFrame).astype(int)

            face_data = [{"name": name, "confidence": (1 - face_distances[best_match_index]),
                          "location": {"top": loc[0], "right": loc[1], "bottom": loc[2], "left": loc[3]}} for
                         name, loc, face_distances, best_match_index
                         in zip(face_names, face_locations, face_distances_list, best_match_indexes_list)]

            return face_data
        else:
            print("No face found")
            return None

    def update_model(self, feedback_data):
        img_path = feedback_data["image_path"]
        if self.load_single_img(img_path):
            index = -1
            if feedback_data["answer"] == "Yes":
                correct_name = feedback_data["prediction"]
                index = self.names.index(correct_name)
            elif feedback_data["answer"] == "No":
                correct_name = feedback_data["correct_name"]
                if correct_name not in self.names:
                    self.names.append(correct_name)
                index = self.names.index(correct_name)

            # Update the encodings and names
            if index >= 0:
                print("Updating model")
                self.encodings[index] = self.encodings[-1]
                self.names[index] = correct_name

            # Save the updated encodings and names to files
            np.save("encodings/{}.npy".format('trainedEncodings'), self.encodings)
            np.save("encodings/{}.npy".format('trainedNames'), self.names)
            print("Model updated")
            return True
        else:
            print("No face found in the image")
            return False

    def delete_image(self, image_path):
        # Get the filename only from the initial file path.
        basename = os.path.basename(image_path)
        (filename, ext) = os.path.splitext(basename)
        # Now delete the image from the folder
        os.remove(image_path)
        # Delete the encoding from the list
        # Load the current encodings and names from files
        self.encodings = np.load("encodings/trainedEncodings.npy", allow_pickle=True)
        self.names = np.load("encodings/trainedNames.npy", allow_pickle=True)
        index = self.names.index(filename)
        del self.names[index]
        del self.encodings[index]
        # Save the updated encodings and names to files
        np.save("encodings/{}.npy".format('trainedEncodings'), self.encodings)
        np.save("encodings/{}.npy".format('trainedNames'), self.names)
        print("Model updated")
        return True

