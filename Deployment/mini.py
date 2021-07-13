##############################################
# Importing the Libraries
##############################################


import streamlit as st
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\AKHIL DUGGIRALA\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
import numpy as np
import cv2
import time

import cam
import math
from scipy import ndimage
import re
from gtts import gTTS
from googletrans import Translator
# import shutil
# import os
# import random
# try:
try:
    from PIL import Image
except ImportError:
    import Image


st.header("Facilitating Surety and Accuracy in the Goods Bought By the Visually Impaired & People With Diminished Vision")


# user_image = cv2.imread('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.png')

path = 'C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.png'


# camera_user()

# st.subheader("Uplaod Image")
# user_image = st.file_uploader("kindly Uplaod Image")

st.write("Please Wait!! 😎😎 till the model recogize the image")

progress = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress.progress(i+1)


st.image(path)

original = Image.open('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG')
st.image(original, use_column_width=True)    
image = cv2.imread('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    


    





path='C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG'
info_text = pytesseract.image_to_string(Image.open(path))
#st.text(info_text)  





recipt_text = info_text.split()
#st.write(recipt_text)



for i in range(len(recipt_text)):
   recipt_text[i] = recipt_text[i].lower()
#st.write(recipt_text)



st.subheader("Query Section")

user_query = st.text_input("Enter your search Query")

st.write(user_query)

count = 0
for i in recipt_text:
  count+=1
  if(i == user_query):
    st.text("match")
    st.subheader("The overall cost of " + user_query + " is:  " + recipt_text[count])
    text_val = recipt_text[count]



f = open("text.txt", "a")
f.write(text_val)
f.close()

f = open("text.txt", "r")
# st.write(f.read())


from gtts import gTTS

import os

f=open('text.txt')
x=f.read()

language='en'

audio=gTTS(text=x,lang=language,slow=False)

audio.save("1.wav")
os.system("1.wav")




st.audio("1.wav")
