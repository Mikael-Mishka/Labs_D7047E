import os
import pathlib
from PIL import Image

def run() -> None:

    normal_in_test: int = 0
    normal_in_val: int = 0

    pneumonia_in_test: int = 0
    pneumonia_in_val: int = 0

    target_dir_path_normal_test: pathlib.Path = pathlib.Path(__file__).parent.parent / "DatasetSplitValTest" / "test" / "NORMAL"
    target_dir_path_normal_val: pathlib.Path = pathlib.Path(__file__).parent.parent / "DatasetSplitValTest" / "val" / "NORMAL"
    target_dir_path_pneumonia_test: pathlib.Path = pathlib.Path(__file__).parent.parent / "DatasetSplitValTest" / "test" / "PNEUMONIA"
    target_dir_path_pneumonia_val: pathlib.Path = pathlib.Path(__file__).parent.parent / "DatasetSplitValTest" / "val" / "PNEUMONIA"

    # Get absolute path of the current directory
    current_directory: pathlib.Path = pathlib.Path(__file__).parent.absolute()

    # Get the DatasetSplitValTest directory
    dataset_split_val_test_directory: pathlib.Path = current_directory.parent / "DatasetValTestCombined"

    # Loop through test and val
    for folder in os.listdir(dataset_split_val_test_directory):

        # Get the path of the folder
        folder_path: pathlib.Path = dataset_split_val_test_directory / folder

        # Loop through the NORMAL and PNEUMONAI folders
        for image_type_folder in os.listdir(folder_path):

            image_type_path: pathlib.Path = folder_path / image_type_folder



            # Loop through images
            for image_name in os.listdir(image_type_path):



                image_path: pathlib.Path = image_type_path / image_name

                # Open the image
                image: Image = Image.open(image_path)

                # Count the number of images in the test and val folders
                if "NORMAL" in image_type_folder:

                    # Check whether to save image to test or val
                    if normal_in_test <= normal_in_val:

                        # Save image to test
                        image.save(target_dir_path_normal_test / image_name)
                        normal_in_test += 1

                    else:
                        # Save image to val
                        image.save(target_dir_path_normal_val / image_name)
                        normal_in_val += 1


                elif "PNEUMONIA" in image_type_folder:

                    # Check whether to save image to test or val
                    if pneumonia_in_test <= pneumonia_in_val:

                        # Save image to test
                        image.save(target_dir_path_pneumonia_test / image_name)
                        pneumonia_in_test += 1

                    else:
                        # Save image to val
                        image.save(target_dir_path_pneumonia_val / image_name)
                        pneumonia_in_val += 1

                else:
                    raise RuntimeError("Invalid image type folder")

    print(f"Normal in test: {normal_in_test}",
          f"Normal in val: {normal_in_val}",
          f"Pneumonia in test: {pneumonia_in_test}",
          f"Pneumonia in val: {pneumonia_in_val}",
          sep="\n")


if __name__ == "__main__":

    run()