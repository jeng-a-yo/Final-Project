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
    prediction = prediction.item()
    
    print(f"Prediction: {histLabels[prediction]}")


transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

numbers = [i for i in range(0, 9+1)]

alphabets = list(sorted(string.ascii_letters))
_alphabets = [chr(i) for i in range(65, 90+1)] + [chr(i) for i in range(97, 122+1)]


symbols = [chr(i) for i in range(33, 47+1)] + \
            [chr(i) for i in range(58, 64+1)] + \
            [chr(91)] + \
            [chr(i) for i in range(93, 96+1)] + \
            [chr(i) for i in range(123, 126+1)]

# numberModel = PaperCNN(in_channels=1, num_classes=len(numbers))
# numberModel.load_state_dict(torch.load("Models/NumberModel.pth"))

# englishModel = LittleFishModel(in_channels=1, num_classes=len(alphabets))
# englishModel.load_state_dict(torch.load("EnglishModel.pth"))

# symbolModel = LittleFishModel(in_channels=1, num_classes=len(symbols))
# symbolModel.load_state_dict(torch.load("SymbolModel.pth"))



# img_path = '5169.png'
# # img_path = 'test_image.png'
# img = Image.open(img_path)  # Ensure the image is in RGB format

# # Apply the transformations
# img = transform(img).unsqueeze(0).float()
# print(img.size())


# Predict(englishModel, alphabets, img)


print(len(numbers), len(alphabets), len(symbols))
