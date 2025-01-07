# Importing necessary libraries
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchsummary import summary
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='PBC classifier')
parser.add_argument('--dataset', type=str,
        help='datasets')
args = parser.parse_args()



# Checking for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# --------------------------------------------------------------------------------------------------------------------------
# Training Phase Below

# Define dataset path
cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(cur_dir, args.dataset)

# Updated transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((360, 360)),                                      # Resize to 360x360           
    transforms.RandomHorizontalFlip(p=0.5),                            # Apply horizontal flip with 50% probability (**Modified**)
    transforms.RandomRotation(degrees=15),                             # Increased rotation range to 15 degrees (**Modified**)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                        saturation=0.3, hue=0.1),                   # Expanded color jitter parameters (**Modified**)
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),             # Added affine transformations (**Added**)
    transforms.ToTensor(),                                             # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])    # Normalize to range [-1, 1] (**Added**)
])

# Load dataset
print("Loading dataset...\n")
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Print the classes (folders) being loaded
print("Classes found in dataset:\n")
for cls, count in zip(dataset.classes, np.bincount([sample[1] for sample in dataset.samples])):
    print(f" - {cls}: {count} images")

# Get dataset indices and labels
indices = np.arange(len(dataset))
labels = [label for _, label in dataset.samples]

# Perform stratified split (75% train, 25% test)
train_indices, test_indices = train_test_split(indices, test_size=0.25, stratify=labels, random_state=42)

# Create subsets for train and test
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Calculate class weights for the training set
train_labels = [dataset.samples[i][1] for i in train_indices]
train_class_counts = np.bincount(train_labels)
train_class_weights = 1.0 / train_class_counts
train_class_weights = torch.tensor(train_class_weights, dtype=torch.float)

# Weighted sampler for training set
train_sample_weights = [train_class_weights[label] for label in train_labels]
train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

# **Removed WeightedRandomSampler for Testing Set** 
# Explanation: Weighted sampling is typically not used for testing as it distorts the class proportions. Testing should reflect real-world distribution.
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)  # Regular shuffling for testing (**Modified**)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)

# Print the number of images per class in the train and test splits
train_class_counts_split = np.bincount([train_labels[i] for i in range(len(train_labels))])
test_labels = [dataset.samples[i][1] for i in test_indices]  # Fixes test_labels being recalculated (**Fixed**)
test_class_counts_split = np.bincount([test_labels[i] for i in range(len(test_labels))])

print("\nNumber of images per class in the training set (75%):\n")
for cls, count in zip(dataset.classes, train_class_counts_split):
    print(f" - {cls}: {count} images")

print("\nNumber of images per class in the testing set (25%):\n")
for cls, count in zip(dataset.classes, test_class_counts_split):
    print(f" - {cls}: {count} images")

print("\nTraining and Testing datasets are prepared with updated WeightedRandomSampler for training and regular shuffling for testing!\n")

# -----------------------------------------------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.3),  # Dropout layer added
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.3),  # Dropout layer added
        )

    def forward(self, x):
        return self.conv_block(x)

class BloodCellResNet(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodCellResNet, self).__init__()
        
        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.3)
        )
        
        # Residual Blocks
        self.res_block1 = ResidualConvBlock(32, 64)
        self.res_block2 = ResidualConvBlock(64, 128)
        self.res_block3 = ResidualConvBlock(128, 256)
        self.res_block4 = ResidualConvBlock(256, 512)
        # self.res_block5 = ResidualConvBlock(512, 1024)
        # Max Pooling (adjusting to ensure feature maps aren't too small)
        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc = nn.Sequential(nn.Linear(512, num_classes))  # Dropout added before output    #, nn.Dropout(0.5)

        # Softmax for multi-class probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        x = self.pool(self.res_block4(x))
        # x = self.pool(self.res_block5(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Define the model
model = BloodCellResNet(num_classes=8)

# Move the model to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the model summary
print("Model Summary:")
summary(model, input_size=(3, 360, 360))  # Input size matches your dataset images

# -----------------------------------------------------------------------------------------------------------------

# Check the size of the first batch of images
for inputs, labels in train_loader:
    print(inputs.shape)  # This should print torch.Size([batch_size, 3, 360, 360])
    break  # Stop after the first batch

# Define the model
model = BloodCellResNet(num_classes=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 8
train_losses = []
train_accuracies = []

# Using mixed precision for faster training (Optional, if using PyTorch >= 1.6)
scaler = torch.amp.GradScaler()

print("\nTraining Process is going on.........\n")

# Training Loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Mixed precision (Updated for newer PyTorch versions)
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Scales the loss, calls backward(), and updates the optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track the loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("\nTraining Complete!\n")

# After training loop, calculate and print the mean loss and accuracy
mean_loss = sum(train_losses) / len(train_losses)
mean_accuracy = sum(train_accuracies) / len(train_accuracies)

print("-----------------------------------------------------------------------------")
print(f"\nAverage Training Loss: {mean_loss:.4f}")
print(f"\nAverage Training Accuracy: {mean_accuracy:.4f}%")
print("\n---------------------------------------------------------------------------\n")

# Plot Training Loss and Accuracy
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the model weights
torch.save(model.state_dict(), "Trained_Blood_Data.pth")
print("Model saved to 'Trained_Blood_Data.pth'\n")

# -----------------------------------------------------------------------------------------------------------------
#  Testing Phase Below


# Set the model to evaluation mode
model.eval()

# # # Load the saved weights (this will be used during testing)
# model.load_state_dict(torch.load("Trained_Blood_Data.pth", weights_only=True))

print("\n\nTesting Process is going on.........\n")

# Initialize variables to track loss and accuracy
test_loss = 0.0
correct_preds_test = 0
total_preds_test = 0
all_preds = []
all_labels = []

class_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'Ig', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

# Use the same loss function and optimizer used during training
criterion = torch.nn.CrossEntropyLoss()

# Evaluate the model on the test dataset
with torch.no_grad():  # No gradient calculations needed during testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)  # Forward pass
        
        # Compute the loss
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)  # Accumulate the loss
        
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds_test += (predicted == labels).sum().item()
        total_preds_test += labels.size(0)
        
        # Collect predictions and labels for classification report
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate average test loss and accuracy
test_loss /= len(test_loader.dataset)  # Average test loss
test_accuracy = (correct_preds_test / total_preds_test ) * 100  # Accuracy

# Print the test loss and accuracy
print(f"\nTesting Loss: {test_loss:.4f}")
print(f"\nTesting Accuracy: {test_accuracy:.4f}%\n")

# Print detailed classification report (precision, recall, F1-score)
print("Classification Report on Testing Set:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 1. Print file names of the test set images for each class
test_class_names = dataset.classes
test_class_files = {cls: [] for cls in test_class_names}

# Retrieve samples of the test dataset
test_samples = [dataset.samples[idx] for idx in test_dataset.indices]

# Collect file names for each class
for file_path, label in test_samples:     
    test_class_name = class_names[label]
    test_class_files[test_class_name].append(file_path)

# Print file names for each class (showing 10 files per class)
for test_class_name, files in test_class_files.items():
    print(f"\nClass: {test_class_name}\n")
    for file in files[:10]:  # Display only first 10 file names for each class
        print(f"  {file}")

# Function to load an image from a file path and predict its class
def classify_image(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Apply the transformations to the image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to GPU

    # Display the input image
    plt.imshow(image.squeeze(0).cpu().permute(1, 2, 0))
    plt.title(f"Input Image")
    plt.axis('off')
    plt.show()

    # Model inference (classification)
    output = model(image)
    _, predicted = torch.max(output, 1)  # Get the predicted class

    # Get the class label (class_names should match the order of the classes used during training)
    predicted_label = predicted.item()

    # The true label can be obtained directly from the folder structure
    true_class_label = class_names.index(image_path.split('/')[-2])  # Extract the folder name as the true label

    # Check if classification is correct
    if predicted_label == true_class_label:
        classification_status = "Yay!!! Correct Classification. I now think I can classify the Blood Cells."
    else:
        classification_status = "Oops!!! Wrong Classification. I think I should train hard to classify the Blood Cells."

    # Print the results
    print(f"\nTrue Class of the Blood Cell: {class_names[true_class_label]}")
    print(f"Predicted Class of the Blood Cell: {class_names[predicted_label]}")
    print(f"\n{classification_status}")
    # print(f"\nClassification Accuracy: {accuracy:.2f}%")
    # print(f"Classification Loss: {loss.item()}")

# Main loop to continuously ask the user to input an image
while True:
    # Prompt the user to input the path to an image and validate the path
    while True:
        image_path = input("\nEnter the path to the Test Image from the file names given above:").strip()
        
        # Check if the path ends with .jpg
        if image_path.lower().endswith('.jpg'):
            # Check if the file exists
            if os.path.isfile(image_path):
                break  # If path is valid, break out of the loop
            else:
                print("\nThe file does not exist. Please Try again.")
        else:
            print("\nThe file must be a .jpg file. Please Try again.")

    # Call the function to classify the image
    classify_image(image_path)

    # Ask the user if they want to classify another image, ensuring proper input
    while True:
        repeat = input("\nDo you want to classify another image? (yes/no): ").strip().lower()
        if repeat == 'yes':
            break
        elif repeat == 'no':
            print("\nExiting the Program. Thank you!\n")
            exit()  # Exit the program
        else:
            print("\nInvalid input. Please type 'yes' or 'no'")