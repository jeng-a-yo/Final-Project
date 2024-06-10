import numpy as np
import pandas as pd
import cv2
from PIL import Image
import string
import matplotlib.pyplot as plt

from myModels import *

def magic_shift(lst, shift):
    n = len(lst)
    new_lst = [None] * n
    for i in range(n):
        new_lst[(i + shift) % n] = lst[i]
    return new_lst

def Predict(model, histLabels, img):
    # Make a prediction
    model.eval()
    prediction = model(img)

    MAGIC_NUMBER_MAP = {10:0, 52:23, 31:4}
    MAGIC_NUMBER = MAGIC_NUMBER_MAP[len(histLabels)]

    
    # Apply softmax to get probabilities
    histValues = F.softmax(prediction, dim=1).detach().numpy().flatten().tolist()
    
    # Create a dictionary of labels and probabilities
    data = dict(zip(magic_shift(histLabels, MAGIC_NUMBER), histValues))
    sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))

    
    #　Plot the bar chart using matplotlib
    # plt.figure(figsize=(10, 6))
    # plt.bar(sorted_data.keys(), sorted_data.values())
    # plt.xlabel('Classes')
    # plt.ylabel('Probability')
    # plt.title('Prediction Probabilities')
    # plt.show()
    
    # Get the predicted class
    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.item()
    
    print(f"Prediction: {histLabels[prediction-MAGIC_NUMBER]}")


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

numbers = [i for i in range(0, 9+1)]

alphabets = list(sorted(string.ascii_letters))
_alphabets = [chr(i) for i in range(65, 90+1)] + [chr(i) for i in range(97, 122+1)]


symbols = [chr(i) for i in range(33, 47+1)] + \
            [chr(i) for i in range(58, 64+1)] + \
            [chr(91)] + \
            [chr(i) for i in range(93, 96+1)] + \
            [chr(i) for i in range(123, 126+1)]

print(len(symbols))

numberModel = NumberModel()
numberModel.load_state_dict(torch.load("Models/NumberModel.pth"))

characterModel = CharacterModel()
characterModel.load_state_dict(torch.load("Models/CharacterModel.pth"))

symbolModel = SymbolModel()
symbolModel.load_state_dict(torch.load("Models/SymbolModel.pth"))



img_path = '8261.png'
img = Image.open(img_path)

# Apply the transformations
# img = CharacterTransform(img).unsqueeze(0).float()
img = SymbolTransform(img).unsqueeze(0).float()
print(img.size())


# Predict(characterModel, alphabets, img)
Predict(symbolModel, symbols, img)