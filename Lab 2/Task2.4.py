
from torchvision.datasets import MNIST





#Deta spliting
import torch
from torchvision import datasets, transforms

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST dataset
])

# Download and load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split the dataset into training and test sets
train_set, val_set = torch.utils.data.random_split(mnist_dataset, [60000, 10000])

# Create data loaders for training and validation sets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)

#traning_data
#valedasun_data
from sklearn.model_selection import train_test_split

# Split the training data into training and validation sets
# Here, 10% of the training data goes to the validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

#test_data

#CNN

    #the CNN model


    #model tranig
        #trening step

        #valedesun step


#adversarial images

    #random imits 







