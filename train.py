import os
from os import walk
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from myModels import *

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories for datasets
data_dirs = ["_NumberDataSet", "_CharacterDataSet", "_SymbolDataSet"]
model_names = ["NumberModel", "CharacterModel", "SymbolModel"]

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001
momentum = 0.9

def measure_time(func):
    """Decorator to measure the execution time of a function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        print(f"[Info] Spent Time: {round(time.time() - start_time, 4)} seconds")
        return
    return wrapper

def train(model, train_loader, val_loader, optimizer, criterion, epochs):
    """Train the model and evaluate on the validation set"""
    model.train()  # Set the model to training mode
    train_acc, train_loss = [], []
    val_acc, val_loss = [], []

    for epoch in range(1, epochs+1):
        train_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training loop
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in progress_bar:
            optimizer.zero_grad()  # Zero the gradients
            predict = model(data.to(device))  # Forward pass
            loss = criterion(predict, target.to(device))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            train_running_loss += loss.item()
            _, predicted = torch.max(predict, 1)
            correct_predictions += (predicted == target.to(device)).sum().item()
            total_predictions += target.size(0)

            progress_bar.set_postfix({'loss': round(loss.item(), 6)})
        
        train_loss.append(train_running_loss / len(train_loader))
        train_acc.append(correct_predictions / total_predictions)

        print(f"[Info] Epoch {epoch}: Training Loss: {round(train_loss[-1], 4)}, Training Accuracy: {round(train_acc[-1], 4)}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation {epoch}")
        with torch.no_grad():  # No need to compute gradients during validation
            for batch_idx, (data, target) in progress_bar_val:
                predict = model(data.to(device))  # Forward pass
                loss = criterion(predict, target.to(device))  # Compute loss
                val_running_loss += loss.item()
                _, predicted = torch.max(predict, 1)
                correct_predictions += (predicted == target.to(device)).sum().item()
                total_predictions += target.size(0)

                progress_bar_val.set_postfix({'val_loss': round(loss.item(), 6)})

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(correct_predictions / total_predictions)

        print(f"[Info] Epoch {epoch}: Validation Loss: {round(val_loss[-1], 4)}, Validation Accuracy: {round(val_acc[-1], 4)}")
        print("----------------------------------------------------------------")

        model.train()  # Set the model back to training mode

    print("[Info] Training completed")
    print("================================================================\n")

    return train_acc, train_loss, val_acc, val_loss

def test(model, test_loader):
    """Test the model on the test set"""
    model.eval()  # Set the model to evaluation mode
    correct_cnt = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for data, target in progress_bar:
            predict = model(data.to(device))  # Forward pass
            answer = predict.argmax(dim=1, keepdim=True)
            correct_cnt += answer.eq(target.to(device).view_as(answer)).sum().item()
            progress_bar.set_postfix({'accuracy': int(100 * correct_cnt / len(test_loader.dataset))})
    
    print(f"[Info] Test Results: Accuracy: {round(100 * correct_cnt / len(test_loader.dataset), 2)}%")

@measure_time
def main():
    """Main function to execute the training and testing pipeline"""

    NumberTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the dataset
    ])

    CharacterTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((64, 64)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the dataset
    ])

    SymbolTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((45, 45)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the dataset
    ])

    transformsList = [NumberTransform, CharacterTransform, SymbolTransform]
    modelsList = [NumberModel, CharacterModel, SymbolModel]
    
    for i in range(len(data_dirs)):

        if i == 0 or i == 2:
            continue

        
        start_time = time.time()

        # Load dataset
        dataset = datasets.ImageFolder(root=data_dirs[i], transform=transformsList[i])

        # Define the sizes for training, validation, and test sets
        train_ratio = 0.7
        val_ratio = 0.15

        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # Split the dataset
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        
        # Build the model
        model = modelsList[i]().to(device)

        # Define the number of epochs

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # Train the model
        print(f"Training {data_dirs[i]}\n")
        train_acc, train_loss, val_acc, val_loss = train(model, train_loader, val_loader, optimizer, criterion, epochs)

        # Evaluate the model
        test(model, test_loader)

        # Save the model
        torch.save(model.state_dict(), f'{model_names[i]}.pth')

        # Plot training and validation loss and accuracy
        plt.figure()
        # Loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        # Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Val Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        # Save figure
        plt.tight_layout()
        plt.savefig(f'{model_names[i]}Graph.png')

        print(f"[Info] Spent Time: {round(time.time() - start_time, 4)} seconds")
        print("================================================================")

main()
