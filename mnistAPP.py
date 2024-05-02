import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# (Blue, Green , Red)

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


def CreateFrame():
    return np.zeros((frameSize, frameSize), dtype=np.uint8)

def ClearFrame():
    global frame
    frame = CreateFrame()

def Recognition():
    feedImg = cv2.resize(frame, (28, 28))
    feedTensor = transforms.ToTensor()(feedImg)
    feedTensor = torch.unsqueeze(feedTensor, dim = 1)
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


model = torch.load("mnistCNN.pt")

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
    


