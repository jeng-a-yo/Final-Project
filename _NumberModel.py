import os
from os import walk
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)

        self.dropout = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

        self.output_layer = nn.Linear(800, 10)


    def forward(self, x):
        

        tensor = self.conv(x)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout(tensor)

        tensor = self.conv2(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout(tensor)

        tensor = torch.flatten(tensor, 1)
        tensor = self.output_layer(tensor)
        output = F.log_softmax(tensor, dim = 1)
        return output


def MeasureTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        print(f"[Info] Spand Time: {round(time.time() - startTime, 4)} seconds")
        return
    return wrapper


def Train(model, trainLoder, epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, epochs+1):
        progress_bar = tqdm(enumerate(trainLoder), total=len(trainLoder), desc=f"Epoch {epoch}")
        for batchIdx, (data, target) in progress_bar:
            optimizer.zero_grad()
            predict = model(data)
            loss = F.nll_loss(predict, target)
            loss.backward()
            optimizer.step()

            if batchIdx % 10 == 0:
                step = batchIdx + 1
                progress_bar.set_postfix({'loss': round(loss.item(), 6)})
                
    print("[Info] Training completed")

def Test(model, testLoder):
    model.eval()
    correctCnt = 0
    with torch.no_grad():
        progress_bar = tqdm(testLoder, desc="Testing")
        for data, target in progress_bar:
            predict = model(data)
            answer = predict.argmax(dim=1, keepdim=True)
            correctCnt += answer.eq(target.view_as(answer)).sum().item()
            progress_bar.set_postfix({'accuracy': int(100 * correctCnt / len(testLoder.dataset))})
    
    print(f"[Info] Test Results: Accuracy: {round(100 * correctCnt / len(testLoder.dataset), 2)}%")


@MeasureTime
def main():

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])

    dataDir = "NumberDataSet"
    batchSize = 64
    epochs = 5


    # Load the dataset
    dataset = datasets.ImageFolder(root=dataDir, transform=transform)

    trainSize = int(0.8 * len(dataset))
    testSize = len(dataset) - trainSize

    trainSet, testSet = random_split(dataset, [trainSize, testSize])

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=32, shuffle=False)

    # Build the model
    model = CNN()


    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    Train(model, trainLoader, epochs)

    # Sava the model
    torch.save(model.state_dict(), 'selfNumberModel.pth')

    # Evaluate the model
    Test(model, testLoader)



main()

