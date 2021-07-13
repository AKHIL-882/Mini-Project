##############################################
# Importing the Libraries
##############################################



import streamlit as st
import nltk
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\AKHIL DUGGIRALA\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
import numpy as np
import cv2


import cam
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


path = 'C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.png'


# camera_user()

# st.subheader("Uplaod Image")
# user_image = st.file_uploader("kindly Uplaod Image")

# st.write("Please Wait!! 😎😎 till the model recogize the image")

progress = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress.progress(i+1)



original = Image.open('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG')
st.image(original, use_column_width=True)    


# path='C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG'
# info_text = pytesseract.image_to_string(Image.open(path))

# st.write(info_text)



##############################################
# Upload Files
##############################################
# user_image = st.file_uploader("kindly Uplaod Image")






##############################################
# Progress Bar
##############################################
# if(user_image):
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.1)
#         progress.progress(i+1)







##############################################
# Displaying Uplaoded Image
##############################################
# st.image(user_image)






##############################################
# Text Detection from Image
##############################################
info_text = pytesseract.image_to_string(Image.open("C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG"))
st.text(info_text)  






##############################################
# Converting Text into array
##############################################
recipt_text = info_text.split()
#st.write(recipt_text)





##############################################
# Converting to lower case
##############################################
for i in range(len(recipt_text)):
   recipt_text[i] = recipt_text[i].lower()
#st.write(recipt_text)






##############################################
# User Speech as Input
##############################################
st.subheader("Query Section")
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Kindly ask you query")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.write("Your Query : ", text)
        except:
            st.write("Please Ask your query correctly")
        return text

##############################################
# Speech Activates when click on button
##############################################
if st.button("Speech as Input"):
    recognized = takecommand()







############################################
# Removing Stopwords
##############################################
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

text_tokens = word_tokenize(recognized)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

#st.write(tokens_without_sw)

filtered_sentence = (" ").join(tokens_without_sw)

recognized = filtered_sentence
# st.write(recognized)



##############################################
# Converting Speech into token of array values
##############################################
arr = recognized.split()
for i in range(len(arr)):
   arr[i] = arr[i].lower()
st.write(arr)







##############################################
# User Speech Mapping
##############################################

#program to print date
for j in range(len(arr)):
  if(arr[j] == 'date'):
      import re
      f = open("text.txt", "r")
      content = f.read()
      pattern = "\d{2}[/-]\d{2}[/-]\d{4}"
      dates = re.findall(pattern, content)
      for date in dates:
          if "-" in date:
              day, month, year = map(int, date.split("-"))
          else:
              day, month, year = map(int, date.split("/"))
          if 1 <= day <= 31 and 1 <= month <= 12:
              # st.write(date)
              text_val = date
      f.close()


#program to print time
for j in range(len(arr)):
  if(arr[j] == 'time'):
    regex = re.compile(r'\d{2}:\d{2}')
    with open('text.txt') as f:
      text_val = regex.findall(f.read())
      #print(text_val)
      def listToString(s):
        str1 = ""
        for ele in s: 
          str1 += ele
        return str1
      text_val =listToString(text_val)
      #st.write(text_val) 
         




#program to find the product prices
for i in range(0,len(recipt_text)):
  for j in range(0,len(arr)):
    if(arr[j] == recipt_text[i]):
      #st.write("match")
      val = arr[j]
      count = 0
      for i in recipt_text:
        count+=1
        if(i == val):
          #st.write("The overall cost of " + val + " is:  " + recipt_text[count])
          text_val = recipt_text[count]
      break;
      if(i==len(arr)):
        i=0;
      if(j==len(recipt_text)):
        j=0;




##############################################
# Copying The Output into a file
##############################################
f = open("text.txt", "w")
f.write(text_val)
f.close()

f = open("text.txt", "r")
# st.write(f.read())








##############################################
# Converting to audio
##############################################

from gtts import gTTS

import os

f=open('text.txt')
x=f.read()

language='en'

audio=gTTS(text=x,lang=language,slow=False)

audio.save("1.wav")
os.system("1.wav")




##############################################
# Making Audio visible to user
##############################################
st.audio("1.wav")


st.balloons()
st.balloons()

st.balloons()
st.balloons()


