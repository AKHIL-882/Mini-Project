##############################################
# Importing the Libraries
##############################################
import streamlit as st
import nltk
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\AKHIL DUGGIRALA\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
import numpy as np
import cv2
import time
import math
from scipy import ndimage
import re
from gtts import gTTS
from googletrans import Translator
import speech_recognition as sr
try:
    from PIL import Image
except ImportError:
    import Image
try:
    os.mkdir("temp")
except:
    pass


import warnings
warnings.filterwarnings("ignore")


##############################################
# Header and Sub Header
##############################################
st.header("Facilitating Surety and Accuracy in the Goods Bought By the Visually Impaired & People With Diminished Vision")
st.subheader("Uplaod Image")




##############################################
# Upload Files
##############################################
user_image = st.file_uploader("kindly Uplaod Image")




path = ""
image = cv2.imread(user_image)



cv2.imwrite(path + "Final_Image.jpg", image)