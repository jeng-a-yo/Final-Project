import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets


import os
from os import walk
import time

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# (Blue, Green , Red)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def CreateFrame():
    return np.zeros((frameSize, frameSize), dtype=np.uint8)

def ClearFrame():
    global frame
    frame = CreateFrame()

def PreprocessImage(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the input size of ResNet18
        transforms.ToTensor(),           # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# def PredictImage():
#     image = PreprocessImage(frame)
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         input_image = PreprocessImage(image)
#         output = model(input_image)
#         _, predicted = torch.max(output, 1)  # Get the index of the class with the highest probability
#         return predicted.item() 

def Recognition():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the input size of ResNet18
        transforms.ToTensor(),           # Convert the image to a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])
    feedImg = transform(frame)

    feedTensor = transforms.ToTensor()(feedImg)
    feedTensor = torch.unsqueeze(feedTensor, dim = 1)

    model = models.resnet18(pretrained=False)
    model.load_state_dict(torch.load('NumberModel.pth'))
    predict = model(feedTensor)

    answer = torch.argmax(predict)
    print(f"Answer: {answer}, {predict}")


def Draw(event, x, y, flags, param):
    global clickFleg, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clickFleg = True
    elif event == cv2.EVENT_LBUTTONUP:
        clickFleg = False
        Recognition()
    elif event == cv2.EVENT_MOUSEMOVE:
        if clickFleg == True:
            cv2.circle(frame, (x, y), 10, 255, -1)


model = torch.load("NumberModel.pth")

frameSize = 512

frame = CreateFrame()
clickFleg = False

cv2.namedWindow("Draw")
cv2.setMouseCallback("Draw", Draw)

while True:

    cv2.imshow("Draw", frame)
    key = cv2.waitKey(33)

    if key == ord('c'):
        ClearFrame()
    elif key == ord('q'):
        break