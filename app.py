import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
from PIL import Image
import string
import os

from myModels import *

st.title = ("Handwritten Character Recognition")

st.markdown("# Handwritten Character Recognition")


selectedModels = st.sidebar.multiselect("Select Model", ["Number Model", "Character Model", "Symbol Model"])

def magic_shift(lst, shift):
    n = len(lst)
    new_lst = [None] * n
    for i in range(n):
        new_lst[(i + shift) % n] = lst[i]
    return new_lst

def Predict(model, histLabels, img):

    model.eval()
    prediction = model(img)

    MAGIC_NUMBER_MAP = {10:0, 52:23, 31:4}
    MAGIC_NUMBER = MAGIC_NUMBER_MAP[len(histLabels)]

    histValues = F.softmax(prediction).detach().numpy().flatten().tolist()
    data = dict(zip(magic_shift(histLabels, MAGIC_NUMBER), histValues))
    st.bar_chart(data)

    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.item()

    st.write(f"Prediction: {histLabels[prediction-MAGIC_NUMBER]}")


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

alphabets = list(sorted(string.ascii_letters))

symbols = [chr(i) for i in range(33, 47+1)] + \
            [chr(i) for i in range(58, 64+1)] + \
            [chr(91)] + \
            [chr(i) for i in range(93, 96+1)] + \
            [chr(i) for i in range(123, 126+1)]


model_folder = "Models"

numberModel = NumberModel(in_channels=1, num_classes=len(numbers))
numberModel.load_state_dict(torch.load(os.path.join(model_folder, "NumberModel.pth")))

characterModel = CharacterModel(in_channels=1, num_classes=len(alphabets))
characterModel.load_state_dict(torch.load(os.path.join(model_folder, "CharacterModel.pth")))

symbolModel = SymbolModel(in_channels=1, num_classes=len(symbols))
symbolModel.load_state_dict(torch.load(os.path.join(model_folder, "SymbolModel.pth")))


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



if st.button("Recognize"):

    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img)
    

    st.image(img, width=width)

    
    if len(selectedModels) == 0:
        st.write("Please select at least one model")
    else:
        if "Number Model" in selectedModels:
            number_input_img = NumberTransform(img).unsqueeze(0).float()
            Predict(numberModel, numbers, number_input_img)
        if "Character Model" in selectedModels:
            character_input_img = CharacterTransform(img).unsqueeze(0).float()
            Predict(characterModel, alphabets, character_input_img)
        if "Symbol Model" in selectedModels:
            img = SymbolTransform(img).unsqueeze(0).float()
            Predict(symbolModel, symbols, img)

    

