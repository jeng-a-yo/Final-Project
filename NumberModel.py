import os
from os import walk
import cv2
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def MeasureTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        print(f"[Info] Spand Time: {round(time.time() - startTime, 4)} seconds")
        return
    return wrapper


@MeasureTime
def main():

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataDir = "NumberDataSet"

    # Load the dataset
    dataset = datasets.ImageFolder(root=dataDir, transform=transform)

    trainSize = int(0.8 * len(dataset))
    testSize = len(dataset) - trainSize

    trainSet, testSet = random_split(dataset, [trainSize, testSize])

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=32, shuffle=False)

    # Build the model
    model = torchvision.models.resnet18(pretrained=True)  # Example model, you can use any model you want

    # Modify the last layer according to your problem
    numFtrs = model.fc.in_features
    numClasses = len(dataset.classes)
    model.fc = torch.nn.Linear(numFtrs, numClasses)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    numEpochs = 10
    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        with tqdm(trainLoader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for inputs, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item() * inputs.size(0)
                tepoch.set_postfix(loss=runningLoss / len(trainSet))
        print("Training Loss: {:.4f}".format(runningLoss / len(trainSet)))

    # Sava the model
    torch.save(model.state_dict(), 'NumberModel.pth')

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testLoader, unit="batch", desc="Testing"):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy on the test set: {:.2f}%'.format(100 * accuracy))



main()

