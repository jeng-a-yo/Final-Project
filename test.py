import numpy as np
import pandas as pd
import cv2
from PIL import Image
import string
import matplotlib.pyplot as plt

from myModels import *

def Predict(model, histLabels, img):
    # Make a prediction
    model.eval()
    prediction = model(img)
    
    # Apply softmax to get probabilities
    histValues = F.softmax(prediction, dim=1).detach().numpy().flatten().tolist()
    # print(histValues)
    
    # Create a dictionary of labels and probabilities
    data = dict(zip(histLabels, histValues))
    for i in data.items():
        print(i)
    
    # Plot the bar chart using matplotlib
    # plt.figure(figsize=(10, 6))
    # plt.bar(data.keys(), data.values())
    # plt.xlabel('Classes')
    # plt.ylabel('Probability')
    # plt.title('Prediction Probabilities')
    # plt.show()
    
    # Get the predicted class
    prediction = torch.argmax(prediction, dim=1)
    print(prediction)
    prediction = prediction.item()
    print(prediction)
    
    print(f"Prediction: {histLabels[prediction]}")


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

numberModel = NumberModel()
numberModel.load_state_dict(torch.load("Models/NumberModel.pth"))

characterModel = CharacterModel()
characterModel.load_state_dict(torch.load("CharacterModel.pth"))

symbolModel = SymbolModel()
symbolModel.load_state_dict(torch.load("SymbolModel.pth"))



img_path = '5169.png'
img = Image.open(img_path)

# Apply the transformations
img = CharacterTransform(img).unsqueeze(0).float()
print(img.size())


Predict(characterModel, alphabets, img)
# Predict(symbolModel, symbols, img)