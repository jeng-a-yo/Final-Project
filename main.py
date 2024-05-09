import streamlit as st
import numpy as np
import pandas as pd
import cv2

from myModels import *

st.title = ("Handwritten Digit Recognition")
# st.markdown("# Handwritten Digit Recognition")

selectedClassifiers = st.sidebar.multiselect("Select Model", ["Number Model", "Character Model", "Symbol Model"])


