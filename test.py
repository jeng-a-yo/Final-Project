import os
from os import walk
import time
import cv2
import numpy as np
import torch
import torchvision.models as models
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


# model = torch.load('NumberModel.pth')

# transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

# image = cv2.imread("0_49.jpg")
# image = transform(image)
# image = image.unsqueeze(0)

# with torch.no_grad():  # 禁用梯度计算
#     output = model(image)

# predicted_class = torch.argmax(output, dim=1).item()
# print("预测结果:", predicted_class)


from PIL import Image


# 加载模型
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define image transformations to match the model's requirements
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 读取并预处理图像
image = Image.open("0_49.jpg")
# image = Image.open("0_280.jpg")
# image = Image.open("1-0001.png")
image = transform(image)
image = image.unsqueeze(0)  # 添加一个批次维度，因为模型期望输入是一个批次

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    output = model(image)

# print("output", output)

# 获取预测结果
predicted_class = torch.argmax(output, dim=1).item()
print("预测结果:", predicted_class)

_, predicted_class = output.max(1)
print(_)
print("Predicted class:", predicted_class.item())
