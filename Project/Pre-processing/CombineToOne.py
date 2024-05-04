"""
Plan of this script:
1. Get the dataset parent folder path.
2. Read all images using loops and pipe
   them into somepath/ModifiedDataset/CombinedToOne
   (makes it easier to work with)
"""

# Important Path and Directory navigating libraries
import os
import pathlib

# Image processing library
from PIL import Image

def run():

    # Define the path to the original dataset. (assume existence)
    dataset_original_path: pathlib.Path = pathlib.Path(__file__).parent.parent / "Datasets"

    # Define the path to the modified dataset. (assume existence)
    dataset_modified_path: pathlib.Path = pathlib.Path(__file__).parent / "ModifiedDatasets" / "CombinedToOne"

    print(dataset_modified_path)

    # Check if the CombinedToOne directory has any files and if it does throw exception
    if len(image_in_combine_path :=os.listdir(dataset_modified_path)) != 0:

        # Go through the images in the directory and count them.
        num_images = 0
        for image in image_in_combine_path:
            num_images += 1

        raise Exception(f"The CombinedToOne directory is not empty. "
                        f"Please clear the directory before running this script "
                        f"(images in directory = {num_images}).")

    """
    IMAGES EXISTS IN THE FOLLOWING PATHS:
    dataset_original_path     /     directory_name     /     last_directory     /     image_name
          (Datasets)          /    (train, test, val)  /   (NORMAL/PNEUMONIA)   /    (image_name)
    """

    for directory_name in os.listdir(dataset_original_path):

        for last_directory in os.listdir(dataset_original_path / directory_name):

            for image_name in os.listdir(dataset_original_path / directory_name / last_directory):

                # Get the image
                image = Image.open(dataset_original_path / directory_name / last_directory / image_name)

                # Save the image to the modified dataset path.
                image.save(dataset_modified_path / image_name)

                image.close()





if __name__ == "__main__":
    run()