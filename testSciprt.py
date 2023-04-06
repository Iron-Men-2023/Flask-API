import os
import glob
import shutil


def rename_and_move_images():
    # Get the list of image files in the static/images directory
    image_files = glob.glob(os.path.join("static/images", "*"))

    # Loop through each image file
    for image_file in image_files:
        # Get the filename without the extension
        file_name, file_ext = os.path.splitext(image_file)

        # Check if the file is an image
        if file_ext.lower() in [".jpg", ".jpeg", ".png", ".gif"]:
            # Create a directory with the same name if it doesn't exist
            if not os.path.exists(file_name):
                os.makedirs(file_name)

            # Get the number of images already in the directory
            num_images = len(glob.glob(os.path.join(file_name, "*")))

            # Create a new file name with img* format
            new_file_name = f"img{num_images + 1}{file_ext}"

            # Move the image to the new directory with the new name
            shutil.move(image_file, os.path.join(file_name, new_file_name))

            print(f"Moved {image_file} to {os.path.join(file_name, new_file_name)}")


if __name__ == "__main__":
    rename_and_move_images()
