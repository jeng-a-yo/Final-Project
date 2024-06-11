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

selected_models = st.sidebar.multiselect("Select Model", ["Number Model", "Character Model", "Symbol Model"])

def magic_shift(lst, shift):
    n = len(lst)
    new_lst = [None] * n
    for i in range(n):
        new_lst[(i + shift) % n] = lst[i]
    return new_lst

def predict(model, hist_labels, img):

    model.eval()
    prediction = model(img)

    magic_number_map = {10: 0, 52: 23, 31: 4}
    magic_number = magic_number_map[len(hist_labels)]

    hist_values = F.softmax(prediction).detach().numpy().flatten().tolist()
    data = dict(zip(magic_shift(hist_labels, magic_number), hist_values))
    st.bar_chart(data)

    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.item()

    st.write(f"Prediction: {hist_labels[prediction - magic_number]}")

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

width = 150
height = 150

numbers = [i for i in range(0, 9 + 1)]

alphabets = list(sorted(string.ascii_letters))

symbols = [chr(i) for i in range(33, 47 + 1)] + \
          [chr(i) for i in range(58, 64 + 1)] + \
          [chr(91)] + \
          [chr(i) for i in range(93, 96 + 1)] + \
          [chr(i) for i in range(123, 126 + 1)]

model_folder = "Models"

number_model = NumberModel(in_channels=1, num_classes=len(numbers))
number_model.load_state_dict(torch.load(os.path.join(model_folder, "NumberModel.pth")))

character_model = CharacterModel(in_channels=1, num_classes=len(alphabets))
character_model.load_state_dict(torch.load(os.path.join(model_folder, "CharacterModel.pth")))

symbol_model = SymbolModel(in_channels=1, num_classes=len(symbols))
symbol_model.load_state_dict(torch.load(os.path.join(model_folder, "SymbolModel.pth")))

number_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the dataset
])

character_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((64, 64)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the dataset
])

symbol_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((45, 45)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the dataset
])

if st.button("Recognize"):

    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img)

    st.image(img, width=width)

    if len(selected_models) == 0:
        st.write("Please select at least one model")
    else:
        if "Number Model" in selected_models:
            number_input_img = number_transform(img).unsqueeze(0).float()
            predict(number_model, numbers, number_input_img)
        if "Character Model" in selected_models:
            character_input_img = character_transform(img).unsqueeze(0).float()
            predict(character_model, alphabets, character_input_img)
        if "Symbol Model" in selected_models:
            symbol_input_img = symbol_transform(img).unsqueeze(0).float()
            img = symbol_transform(img).unsqueeze(0).float()
            predict(symbol_model, symbols, img)
