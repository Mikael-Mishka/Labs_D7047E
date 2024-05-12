import os
import pathlib
from PIL import Image
import torch
import torchvision.transforms

def test():

    # Get absolute a specific image
    test_img_path = pathlib.Path(__file__).parent.parent / "DatasetSplitValTest" / "test" / "NORMAL" / "IM-0001-0001.jpeg"

    img = Image.open(test_img_path)

    transform = train_augmentation((500, 256))

    # Import DataLoader
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.dataset import TensorDataset

    img_tensor = transform(img)

    # Make a TensorDataset
    dataset = TensorDataset(img_tensor, torch.LongTensor([0]))

    # Make a DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for i, (image, target) in enumerate(loader):
        print(image.shape)
        print(target.shape)

    print(transform(img).shape)


def train_augmentation(dataset_size: tuple[int, int]) -> torchvision.transforms.Compose:

    # Augmentation
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(dataset_size),
        torchvision.transforms.Grayscale(num_output_channels=1)
    ])


"""
This function will apply the transform to all images in the dataset.
"""
def transform_dataset(original_dataset_path: pathlib.Path,
                      transform_dir_path: pathlib.Path) -> None:

    """
    ORIGINAL DATASET STRUCTURE:
        - original_dataset_path
            - test
                - NORMAL
                - PNEUMONIA
            - train
                - NORMAL
                - PNEUMONIA
            - val
                - NORMAL
                - PNEUMONIA
    """

    if not transform_dir_path.exists():
        os.makedirs(transform_dir_path)

    new_size = (256, 256)

    # Keep track of the labels in order
    labels_train = []
    labels_val = []
    labels_test = []

    # Keep track of the images in order
    train_images = []
    val_images = []
    test_images = []

    # Loop through the sub-directories
    for sub_directory in os.listdir(original_dataset_path):

        sub_directory_path: pathlib.Path = original_dataset_path / sub_directory

        # Loop through the sub-sub-directories (NORMAL and PNEUMONIA)
        for image_type_sub_dir in os.listdir(sub_directory_path):

            image_type_sub_dir_path: pathlib.Path = sub_directory_path / image_type_sub_dir

            # Loop through the images
            for image in os.listdir(image_type_sub_dir_path):

                image_path: pathlib.Path = image_type_sub_dir_path / image

                # Open the image
                img: Image = Image.open(image_path)

                # Apply the transform
                transformed_img: Image = train_augmentation(new_size)(img)

                # Save the transformed image
                transformed_img.save(transform_dir_path / sub_directory / image_type_sub_dir / image)

def run() -> None:

    # Get the "DatasetResizedGrayscaled" directory
    abs_path = pathlib.Path(__file__).parent.absolute()

    # The dataset_directory_path
    target_dir_path = abs_path.parent / "DatasetResizedGrayscaled"

    # Get the "DatasetSplitValTest" directory
    original_dataset_path = abs_path.parent / "DatasetSplitValTest"

    print("Transforming the dataset...",
          f"From: {original_dataset_path}",
          f"To: {target_dir_path}",
          sep="\n")

    transform_dataset(original_dataset_path, target_dir_path)


run()