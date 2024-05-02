import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, tensor):
        tensor = F.relu(self.conv1(tensor))
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout1(tensor)

        tensor = F.relu(self.conv2(tensor))
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout1(tensor)

        tensor = F.relu(self.conv3(tensor))
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout2(tensor)

        tensor = torch.flatten(tensor, 1)

        tensor = F.relu(self.fc1(tensor))
        tensor = self.dropout2(tensor)
        tensor = self.fc2(tensor)

        return F.log_softmax(tensor, dim=1)

def OrdStringsConvertion(char):
    charToASCII_Dict = {"dot": 46, "minus": 45, "plus": 43, "slash": 47}
    if char in charToASCII_Dict.keys():
        return charToASCII_Dict[char]
    else:
        return ord(char)

def ConvertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


#(28, 28, 1)
def LoadMnist():

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
    ])

    trainDataSet = datasets.MNIST("./data",
                            train = True,
                            download = True,
                            transform = transform
    )

    print("MNIST dataset loading completed.")


    return trainDataSet

# (64, 64, 3)
def LoadCurated():

    dataDict = {}

    for charCode in os.listdir("curated"):
        directory = f"curated/{charCode}"
        if os.path.isdir(directory):
            charImgList = []

            for fileName in os.listdir(directory):
                if fileName.endswith(".png"): 
                    filePath = os.path.join(directory, fileName)
                    image = cv2.imread(filePath)
                    if image is not None:
                        charImgList.append(image)
        dataDict[charCode] = charImgList

    print("Curated dataset loading completed.")

    return dataDict


# (28, 28, 3)
def LoadBhmsds():
    directory = "bhmsds/symbols"
    charList = sorted(list(set([fileName[:fileName.index('-')] for fileName in os.listdir(directory) if fileName.endswith(".png")])))
    dataDict = {}

    for char in charList:
        dataDict[OrdStringsConvertion(char)] = []


    
    for fileName in os.listdir(directory):
        if fileName.endswith(".png"):
            filePath = os.path.join(directory, fileName)
            image = cv2.imread(filePath)
            if image is not None:
                dataDict[OrdStringsConvertion(fileName[:fileName.index('-')])].append(image)


    print("Bhmsds dataset loading completed.")
    return dataDict


@MeasureTime
def main():

    MnistData = LoadMnist()
    CuratedData = LoadCurated()
    BhmsdsData = LoadBhmsds()

    

    # print(type(MnistData))
    # print(type(CuratedData))

if __name__ == "__main__":
    main()