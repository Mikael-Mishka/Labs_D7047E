# IMPORTANT PATH LIBRARIES: os, pathlib
import os
import pathlib
from pathlib import Path
from typing import List
from PIL import Image

"""
The point of this class is the following:
    1. Get a parent folder for a dataset
    2. Statistics for the dataset (width and height mean,
                                   standard deviation.)
    3. Get the number of classes in the dataset.
    4. Get the number of images per class in the dataset.
    5. Do the bulk of the work in the __init__ method.

"""
class DatasetStatisticsManager:

    def __init__(self, dataset_directory_path: Path, num_of_classes: int) -> None:

        self.num_of_classes: int = num_of_classes
        self.dataset_path: Path = dataset_directory_path

        # Same method for any instance of the class for all datasets.
        dataset_directories: list[str] = DatasetStatisticsManager.get_sub_directories(dataset_directory_path)

        # Conditions which will prevent us from continuing.
        dataset_directories_fail_conditions: list[bool] = [
            "train" not in dataset_directories,
            "test" not in dataset_directories,
            "val" not in dataset_directories,
            len(dataset_directories) != 3
        ]

        # Make sure dataset has the the correct length and names structure.
        if any(dataset_directories_fail_conditions):
            raise RuntimeError("Dataset must have 3 sub-directories: train, test, and validation."
                               "In addition must have the correct names for the sub-directories.")

        # Get the width and height mean of the images in the directories
        self.train_mean_width, self.train_mean_height = DatasetStatisticsManager.get_mean_width_and_height_of_images_in_directory(dataset_directory_path / "train")
        self.test_mean_width, self.test_mean_height = DatasetStatisticsManager.get_mean_width_and_height_of_images_in_directory(dataset_directory_path / "test")
        self.val_mean_width, self.val_mean_height = DatasetStatisticsManager.get_mean_width_and_height_of_images_in_directory(dataset_directory_path / "val")

        # Get the standard deviation of the images in the directories
        self.train_SD_width, self.train_SD_height = DatasetStatisticsManager.get_standard_deviation_of_images_in_directory(dataset_directory_path / "train",
                                                                                                                                             self.train_mean_width,
                                                                                                                                             self.train_mean_height)
        # SD = Standard Deviation (test)
        self.test_SD_width, self.test_SD_height = DatasetStatisticsManager.get_standard_deviation_of_images_in_directory(dataset_directory_path / "test",
                                                                                                                  self.test_mean_width,
                                                                                                                  self.test_mean_height)
        # SD = Standard Deviation (val)
        self.val_SD_width, self.val_SD_height = DatasetStatisticsManager.get_standard_deviation_of_images_in_directory(dataset_directory_path / "val",
                                                                                                                 self.val_mean_width,
                                                                                                              self.val_mean_height)

        # Get smallest width and height across all datasets.
        self.smallest_width, self.smallest_height = DatasetStatisticsManager.get_smallest_width_and_height(dataset_directory_path)

        # Print the statistics of the dataset.
        print(f"Dataset Statistics: ",
              f"Number of classes       = {self.num_of_classes}",
              f"---------------------------------",
              f"Train:      Mean Width  = {self.train_mean_width}",
              f"Train:      Mean Height = {self.train_mean_height}",
              f"Train:      SD Width    = {self.train_SD_width}",
              f"Train:      SD Height   = {self.train_SD_height}",
              f"---------------------------------",
              f"Test:       Mean Width  = {self.test_mean_width}",
              f"Test:       Mean Height = {self.test_mean_height}",
              f"Test:       SD Width:   = {self.test_SD_width}",
              f"Test:       SD Height   = {self.test_SD_height}",
              f"---------------------------------",
              f"Validation: Mean Width  = {self.val_mean_width}",
              f"Validation: Mean Height = {self.val_mean_height}",
              f"Validation: SD Width    = {self.val_SD_width}",
              f"Validation: SD Height   = {self.val_SD_height}",
              f"---------------------------------",
              f"Dataset:    Smallest Width: {self.smallest_width} (over all images)",
              f"Dataset:    Smallest Height: {self.smallest_height} (over all images)",
              f"---------------------------------",
              sep="\n")





    @staticmethod
    def get_smallest_width_and_height(dataset_directory_path: Path) -> List[int]:
        smallest_widths: list[int] = []
        smallest_heights: list[int] = []

        """
        - Dataset directory path (path)
        -- path / [train, test, val] (path2)
        --- path2 / image_name (path3)
        """

        for current_directory_path in os.listdir(dataset_directory_path):

            for dataset_type_name in os.listdir(dataset_directory_path / current_directory_path):

                for image_name in os.listdir(dataset_directory_path / current_directory_path / dataset_type_name):

                    image: Image = Image.open(dataset_directory_path / current_directory_path / dataset_type_name / image_name)

                    width, height = image.size

                    smallest_widths.append(width)
                    smallest_heights.append(height)

                    image.close()



        return (min(smallest_widths), min(smallest_heights))


    @staticmethod
    def get_standard_deviation_of_images_in_directory(
            directory_path: Path,
            mean_width: float,
            mean_height: float)\
            -> tuple[float, float]:

        """
            Step 1: Find the mean.
            Step 2: Subtract the mean from each score.
            Step 3: Square each deviation.
            Step 4: Add the squared deviations.
            Step 5: Divide the sum by the number of scores.
            Step 6: Take the square root of the result from Step 5.
        """

        # Get the width and height of the images in the directory.
        widths: list[float] = []
        heights: list[float] = []

        # Deviation squared
        deviation_squared_widths: list[float] = []
        deviation_squared_heights: list[float] = []

        # for directory in os.listdir(directory_path):
        for current_directory_path in os.listdir(directory_path):

            # We will the file names and use PIL image to get the width and height.
            for image_name in os.listdir(directory_path / current_directory_path):

                # Prepare the image for processing.
                image: Image = Image.open(directory_path /current_directory_path / image_name)

                # Get the width and height of the image.
                width, height = image.size

                # Append the width and height to the lists.
                widths.append(width)
                heights.append(height)

                # Get the deviation squared for the width and height.
                deviation_squared_widths.append((width - mean_width) ** 2)
                deviation_squared_heights.append((height - mean_height) ** 2)

        # Calculate the standard deviation for the width and height.
        standard_deviation_width = (sum(deviation_squared_widths) / len(deviation_squared_widths)) ** 0.5
        standard_deviation_height = (sum(deviation_squared_heights) / len(deviation_squared_heights)) ** 0.5

        return standard_deviation_width, standard_deviation_height


    @staticmethod
    def get_mean_width_and_height_of_images_in_directory(directory_path: Path) -> tuple[float, float]:

        widths: list[float] = []
        heights: list[float] = []

        for current_directory_path in os.listdir(directory_path):

            # We will the file names and use PIL image to get the width and height.
            for image_name in os.listdir(directory_path / current_directory_path):

                # Prepare the image for processing.
                image: Image = Image.open(directory_path / current_directory_path / image_name)

                # Get the width and height of the image.
                width, height = image.size

                # Append the width and height to the lists.
                widths.append(width)
                heights.append(height)

        # Get the mean of the widths and heights.
        mean_width: float = sum(widths) / len(widths)
        mean_height: float = sum(heights) / len(heights)

        return mean_width, mean_height


    @staticmethod
    def get_sub_directories(directory_path: Path) -> list[str]:
        return [directory for directory in os.listdir(directory_path)]