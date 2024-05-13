import pathlib
import os
from PIL import Image
import torchvision.transforms.functional as F
import torch
import pickle

# Import TensorDataset
from torch.utils.data import TensorDataset


def statistics_of_tensors(list_of_tensors):
    smallest_width = 1000000
    smallest_height = 1000000

    tensors_mean_width = 0
    tensors_mean_height = 0

    amount_of_tensors = len(list_of_tensors)

    standard_deviation_width = 0
    standard_deviation_height = 0

    for tensor in list_of_tensors:

        # Add the width and height of the tensor
        tensors_mean_width += tensor.size(0)
        tensors_mean_height += tensor.size(1)

        if tensor.size(0) < smallest_width:
            smallest_width = tensor.size(0)

        if tensor.size(1) < smallest_height:
            smallest_height = tensor.size(1)

    # Calculate the mean width and height
    tensors_mean_width /= amount_of_tensors
    tensors_mean_height /= amount_of_tensors

    # Calculate the standard deviation
    for tensor in list_of_tensors:
        standard_deviation_width += (tensor.size(0) - tensors_mean_width) ** 2
        standard_deviation_height += (tensor.size(1) - tensors_mean_height) ** 2

    # Devide by the amount of scores
    standard_deviation_width /= amount_of_tensors
    standard_deviation_height /= amount_of_tensors

    # Get final standard deviation
    standard_deviation_width = standard_deviation_width ** 0.5
    standard_deviation_height = standard_deviation_height ** 0.5

    return smallest_width, smallest_height, tensors_mean_width, tensors_mean_height, standard_deviation_width, standard_deviation_height


"""
1. Vi får veta orignella bildernas map
2. Vi får veta vilken mapp de transformerade bilderna ska.
3. Vi får veta vilken storlek bilderna ska ha.
"""
def decolorize_and_rescale_images(original_images_path: str,
                                  transformed_images_path: str,
                                  size: tuple[int, int]) -> None:

    # Get all the images in the original_images_path
    original_images = os.listdir(original_images_path)

    num_of_bacterial = 0
    num_of_viral = 0
    num_of_normal = 0

    # Loop through all the images
    for image in original_images:
        # Open the image
        img = Image.open(original_images_path + image)

        # Convert the image to grayscale
        img = img.convert('L')

        # Viral images
        if ("VIRAL" in transformed_images_path) and ("virus" in image):
            num_of_viral += 1
            img.resize(size).save(transformed_images_path + image)
            continue
        # Bacterial images
        if ("BACTERIAL" in transformed_images_path) and ("bacteria" in image):
            num_of_bacterial += 1
            img.resize(size).save(transformed_images_path + image)
            continue

        if ("NORMAL" in transformed_images_path) and ("NORMAL" in image):
            num_of_normal += 1
            img.resize(size).save(transformed_images_path + image)
            continue


    print(sum([num_of_normal, num_of_bacterial, num_of_viral]),
          f"BACTERIAL: {num_of_bacterial}",
          f"VIRAL: {num_of_viral}",
          f"NORMAL: {num_of_normal}", sep="\n")


def get_train_data_and_labels(normal_path, pneumonia_path):
    normal = create_tensors_from_images(normal_path)
    pneumonia = create_tensors_from_images(pneumonia_path)
    
    return torch.cat([normal, pneumonia]), len(normal), len(pneumonia)



def create_tensors_from_images(images_path: str) -> list[torch.Tensor]:
    # Get all the images in the images_path
    images = os.listdir(images_path)

    # Create a list to store the tensors
    tensors: list[torch.tensor] = []

    # Loop through all the images
    for image in images:
        # Open the image
        img = Image.open(images_path + image)

        # Convert the image to a tensor
        tensor = F.to_tensor(img)

        # Add the tensor to the list
        tensors.append(tensor)

    # Return torch.stack of tensors list
    return torch.stack(tensors)


def main():
    print("Decolorizing and rescaling images...",
          "DONE!!!", sep="\n")

    original_image_paths = {
        "NORMAL": "/home/convergent/PycharmProjects/Labs D7047E/Project/Datasets/train/NORMAL/",
        "PNEUMONIA": "/home/convergent/PycharmProjects/Labs D7047E/Project/Datasets/train/PNEUMONIA/"
    }

    transformed_image_paths = {
        "NORMAL": "/home/convergent/PycharmProjects/Labs D7047E/Project/Pre-processing/OneColorChannelImages/train/NORMAL/",
        "BACTERIAL": "/home/convergent/PycharmProjects/Labs D7047E/Project/Pre-processing/OneColorChannelImages/train/BACTERIAL/",
        "VIRAL": "/home/convergent/PycharmProjects/Labs D7047E/Project/Pre-processing/OneColorChannelImages/train/VIRAL/"
    }

    # These does a decolorization and rescaling of the images.
    decolorize_and_rescale_images(original_image_paths["NORMAL"],
                                  transformed_image_paths["NORMAL"],
                                  (600, 500))

    decolorize_and_rescale_images(original_image_paths["PNEUMONIA"],
                                  transformed_image_paths["VIRAL"],
                                  (600, 500))

    decolorize_and_rescale_images(original_image_paths["PNEUMONIA"],
                                  transformed_image_paths["BACTERIAL"],
                                  (600, 500))


"""
1. (Halfway done) Pre-process (scaling, sorting into folders, decolorizing)
2. Create tensor chunks using pickle (memory management don't run out of RAM)
3. Define the model(s).
4. Training
5. Testing
6. ???
"""

main()
