"""
The point of this file is to create several resized versions
of the dataset to test the performance of the models on different
resolutions.
----------------------------------------------------------------------------------------
1. Make a function which can provided a dataset directory path and a target
   directory path resize all the images in the dataset directory path and
   append them to the target directory path.
2. Make some code to make a few resized versions of the dataset.
"""

# Import file management libraries
import pathlib
import os

# Import image processing libraries
from PIL import Image


# Define the function to resize the images in the dataset directory.
def resize_images_in_directory(dataset_directory_path: str | pathlib.Path,
                               target_directory_path: str | pathlib.Path, variant: str) -> None:
    """
    Resize all the images in the dataset directory
    and append them to the target directory.

    ASSUME: 'dataset_directory_path' has no subdirectories.
    """

    width: int = 0
    height: int = 1

    # Smallest width and height of the ENTIRE dataset
    if variant == "smallest":
        width, height = 384, 127

    # This uses the train mean obtained earlier
    elif variant == "mean":
        width, height = 1320, 968

    # Get the list of image file names in the dataset directory.

    for image_file_name in os.listdir(dataset_directory_path):

        # Get the image
        image: Image = Image.open(dataset_directory_path / image_file_name)

        # Resize the image
        resized_image: Image = image.resize((width, height))

        # Save the resized image to the target directory
        resized_image.save(target_directory_path / image_file_name)

        # Close the image
        image.close()


def run() -> None:
    """
    Defines the dataset directory paths in a dictionary.
        "variant_name": "dataset_directory_path",
        ...
    """

    # Get the absolute path of the current file.
    abs_path: pathlib.Path = pathlib.Path(__file__).parent.absolute()

    # Get the ModifiedDatasets directory path.
    modified_datasets_path: pathlib.Path = abs_path / "ModifiedDatasets"

    # Define the dataset directory of which you want to resize.
    dataset_directory_path: pathlib.Path = modified_datasets_path / "DecolorizeCombineToOne"

    # Define the dataset directories lookup table.
    dataset_directories_lookup_table: dict[str, str | pathlib.Path] = {
        "smallest": modified_datasets_path / "SmallestWidthHeightDatasets",
        "mean": modified_datasets_path / "MeanWidthHeightDatasets",
    }

    # Utilize the lookup table to resize the images in the dataset directory
    # and append them to the target directory.
    for variant_name, target_directory_path in dataset_directories_lookup_table.items():
        print(target_directory_path)
        # Check if the target directory already has files
        if len(os.listdir(target_directory_path)) != 0:
            raise Exception(f"Target directory '{target_directory_path}' is not empty.")

        resize_images_in_directory(dataset_directory_path, target_directory_path, variant_name)



# Runs 'run' if this file is run.
if __name__ == "__main__":
    run()
