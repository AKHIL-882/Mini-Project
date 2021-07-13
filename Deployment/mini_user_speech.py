##############################################
# Importing the Libraries
##############################################

import nltk

import streamlit as st
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
# import shutil
# import os
# import random
# try:
try:
    from PIL import Image
except ImportError:
    import Image


st.header("Facilitating Surety and Accuracy in the Goods Bought By the Visually Impaired & People With Diminished Vision")


st.subheader("Uplaod Image")
user_image = st.file_uploader("kindly Uplaod Image")
if(user_image):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)


st.image(user_image)





try:
    os.mkdir("temp")
except:
    pass



info_text = pytesseract.image_to_string(Image.open(user_image))
#st.text(info_text)  



recipt_text = info_text.split()
#st.write(recipt_text)



for i in range(len(recipt_text)):
   recipt_text[i] = recipt_text[i].lower()
#st.write(recipt_text)



st.subheader("Query Section")







user_query = st.text_input("Enter your search Query")


from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


text_tokens = word_tokenize(user_query)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

st.write(tokens_without_sw)

filtered_sentence = (" ").join(tokens_without_sw)
st.write(filtered_sentence)



# st.write(user_query)

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
