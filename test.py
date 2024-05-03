import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

from myModels import *



def Recognition(imagePath):
    feedImg = cv2.imread(imagePath)
    feedImg = cv2.resize(feedImg, (28, 28))
    feedTensor = transforms.ToTensor()(feedImg)
    feedTensor = torch.unsqueeze(feedTensor, dim = 1)
    print(feedImg.shape)
    print(feedTensor.shape)
    predict = model(feedTensor)
    print(predict)
    answer = torch.argmax(predict)
    print(f"Answer: {answer}")

model = PaperCNN(10)
model.load_state_dict(torch.load("NumberModel.pth"))
model.eval()

imagePath = "1-0001.png"
# imagePath = "0_280.jpg"

Recognition(imagePath)


