import os
import cv2
import numpy as np
from google.cloud import storage
from firebase_admin import credentials, initialize_app, firestore


class FirebaseImageDownloader:
    def __init__(self, firebase_json_key, firebase_bucket_name, download_folder='static/images/'):
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
                    # save the image temporarily
                    cv2.imwrite("imagesTest/temp.jpg", image)

                    # save image to folder with the user's name and name img1, img2,
                    # etc. based on the last number of the last image saved. If there are two images that look alike
                    # then delete one of the images in the folder
                    # Check if the folder exists
                    if not os.path.exists(self.download_folder + user):
                        os.makedirs(self.download_folder + user)
                    # Check if the folder is empty
                    if len(os.listdir(self.download_folder + user)) == 0:
                        cv2.imwrite(self.download_folder + user + "/img1.jpg", image)
                        print('image saved')
                    else:
                        # Get the list of images in the folder
                        files = os.listdir(self.download_folder + user)
                        # Get the last image in the folder
                        last_image = files[-1]
                        # Get the last number in the image name
                        last_number = int(last_image[3:-4])
                        # Check if the image is similar to the last image in the folder
                        # if self.check_image_similarity(self.download_folder + user + "/" + last_image,
                        #                                "imagesTest/temp.jpg"):
                        #     continue
                        # else:
                        # Save the image with the next number
                        cv2.imwrite(self.download_folder + user + "/img" + str(last_number + 1) + ".jpg", image)
                        print('image saved')
                except KeyError:
                    continue

    def check_image_similarity(self, image1, image2):
        print('image1', image1)
        print('image2', image2)
        # Check if the two images are similar
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        # make sure the images are the same size
        image1 = cv2.resize(image1, (100, 100))
        image2 = cv2.resize(image2, (100, 100))
        difference = cv2.subtract(image1, image2)
        b, g, r = cv2.split(difference)
        print('b', cv2.countNonZero(b))
        print('g', cv2.countNonZero(g))
        print('r', cv2.countNonZero(r))
        # Check if the images are the same or very similar
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return True
        else:
            return False

    def delete_similar_images(self, folder):
        # Delete similar images
        files = os.listdir(folder)
        for i in range(len(files)):
            # check if there is more than one image in the folder
            if len(files) == 1:
                break
            for j in range(i + 1, len(files)):
                if self.check_image_similarity(os.path.join(folder, files[i]), os.path.join(folder, files[j])):
                    os.remove(os.path.join(folder, files[j]))


if __name__ == "__main__":
    firebase_json_key = 'omnilens-d5745-firebase-adminsdk-rorof-6a932e9332.json'
    firebase_bucket_name = 'omnilens-d5745.appspot.com'
    downloader = FirebaseImageDownloader(firebase_json_key, firebase_bucket_name)
    downloader.download_blobs_in_folder('images/Avatar')
