"""
PLAN:
1. Get the CombineToOne dataset directory path.
2. Decolorize all the images in the dataset.
"""

# Import pathlib, os for file manipulation.
import pathlib
import os

# Import Image from PIL for image manipulation.
from PIL import Image

def run() -> None:

    # Define the path to the CombineToOne directory.
    abs_path: pathlib.Path = pathlib.Path(__file__).parent.absolute()

    # CombineToOne directory path.
    combine_to_one_directory_path: pathlib.Path = (abs_path /
                                                   "ModifiedDatasets" /
                                                   "CombinedToOne")

    # Path to save the decolorized images.
    decolorized_directory_path: pathlib.Path = (abs_path /
                                                "ModifiedDatasets" /
                                                "DecolorizeCombineToOne")

    # If the decolorized directory is not empty. Throw an error.
    if len(os.listdir(decolorized_directory_path)) != 0:
        raise FileExistsError("The directory is not empty. Please empty it before running this script.")

    # Go through all the images in the "combine_to_one_directory_path".
    for image_path in os.listdir(combine_to_one_directory_path):

        # Open the image.
        image = Image.open(combine_to_one_directory_path / image_path)

        # Convert the image to grayscale.
        image = image.convert("L")

        # Save the image to the "decolorized_directory_path".
        image.save(decolorized_directory_path / image_path)

        # Close the image.
        image.close()



# If you run this intended as a script, it will run the 'run' function.
if __name__ == "__main__":
    run()
