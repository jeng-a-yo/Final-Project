import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import string

from myModels import *

st.title = ("Handwritten Digit Recognition")

st.markdown("# Handwritten Digit Recognition")


selectedModels = st.sidebar.multiselect("Select Model", ["Number Model", "Character Model", "Symbol Model"])


def LoadModel():
    numberModel = torch.load("NumberModel.pth")
    englishModel = torch.load("EnglishModel.pth")
    symbolModel =torch.load("SymbolModel.pth")

    return numberModel, englishModel, symbolModel

def Predict(model, histLabels, img):

    prediction = model(img)
    histValues = F.softmax(prediction).detach().numpy().flatten().tolist()
    data = dict(zip(histLabels, histValues))
    st.bar_chart(data)

    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.item()

    st.write(f"Prediction: {histLabels[prediction]}")


canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)


transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

width = 150
height = 150

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

print("complete loading model")


if st.button("Recognize"):

    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img)
    

    st.image(img, width=width)

    img = transform(img).unsqueeze(0).float()

    
    if len(selectedModels) == 0:
        st.write("Please select at least one model")
    else:
        if "Number Model" in selectedModels:
            Predict(numberModel, numbers, img)
        if "Character Model" in selectedModels:
            Predict(englishModel, alphabets, img)
        if "Symbol Model" in selectedModels:
            Predict(symbolModel, symbols, img)

    

