import os
from os import walk
import cv2
import time
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from myModels import *


def MeasureTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        print(f"[Info] Spand Time: {round(time.time() - startTime, 4)} seconds")
        return
    return wrapper


def Train(model, trainLoder, optimizer, criterion, epochs):

    model.train()
    for epoch in range(1, epochs+1):
        progressBar = tqdm(enumerate(trainLoder), total=len(trainLoder), desc=f"Epoch {epoch}")

        for batchIdx, (data, target) in progressBar:
            optimizer.zero_grad()
            predict = model(data)
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()


            progressBar.set_postfix({'loss': round(loss.item(), 6)})
                
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

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)

    # Build the model
    # model = CNN()
    # model = AlexNet(10)
    model = PaperCNN(10)


    # model = models.vgg16()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    Train(model, trainLoader, optimizer, criterion, epochs)

    # Sava the model
    torch.save(model.state_dict(), 'NumberModel.pth')

    # Evaluate the model
    Test(model, testLoader)



main()

