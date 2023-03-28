from download_images_from_firebase import FirebaseImageDownloader
from simple_facerec import RecognitionHelper


def train():
    recognizer = RecognitionHelper()
    recognizer.train_on_images("static/images")


if __name__ == "__main__":
    firebase_json_key = 'omnilens-d5745-firebase-adminsdk-rorof-df461ea39d.json'
    firebase_bucket_name = 'omnilens-d5745.appspot.com'
    downloader = FirebaseImageDownloader(firebase_json_key, firebase_bucket_name)
    downloader.download_blobs_in_folder('images/Avatar')
    train()
