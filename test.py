import numpy as np
import pandas as pd
import cv2
from PIL import Image
import string
import matplotlib.pyplot as plt



from myModels import *

def Predict(model, histLabels, img):
    # Make a prediction
    prediction = model(img)
    
    # Apply softmax to get probabilities
    histValues = F.softmax(prediction, dim=1).detach().numpy().flatten().tolist()
    
    # Create a dictionary of labels and probabilities
    data = dict(zip(histLabels, histValues))
    
    # Plot the bar chart using matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(data.keys(), data.values())
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.show()
    
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

alphabets = list(string.ascii_letters)

symbols = [chr(i) for i in range(33, 47+1)] + \
            [chr(i) for i in range(58, 64+1)] + \
            [chr(91)] + \
            [chr(i) for i in range(93, 96+1)] + \
            [chr(i) for i in range(123, 126+1)]

numberModel = PaperCNN(in_channels=1, num_classes=len(numbers))
numberModel.load_state_dict(torch.load("NumberModel.pth"))

englishModel = PaperCNN(in_channels=1, num_classes=len(alphabets))
englishModel.load_state_dict(torch.load("EnglishModel.pth"))

symbolModel = PaperCNN(in_channels=1, num_classes=len(symbols))
symbolModel.load_state_dict(torch.load("SymbolModel.pth"))


img_path = 'test_image.png'
img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format

# Apply the transformations
img = transform(img).unsqueeze(0).float()

Predict(englishModel, alphabets, img)