import streamlit as st

st.title("welcome")


from PIL import Image
import cv2 
import numpy as np
import pytesseract
import os
import time
import glob
import os


from gtts import gTTS
from googletrans import Translator

try:
    os.mkdir("temp")
except:
    pass
st.title("Text to speech")
translator = Translator()
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\AKHIL DUGGIRALA\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
def photo():
	st.header("Thresholding, Edge Detection and Contours")
	    
	# if st.button('See Original Image of Tom'):
	original = Image.open('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/2.jpg')
	st.image(original, use_column_width=True)    
	image = cv2.imread('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/2.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	 
if st.button('photo'):
    photo()
    #st.write('result: %s' % result)


    
path='C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/2.jpg'
info_text = pytesseract.image_to_string(Image.open(path))
st.text(info_text)  

recognized = "Please! tell cost of watches"
def Convert(string):
	li = list(string.split(" "))
	return li
li = Convert(recognized)
articles = {'a','of','only','for','an','the','please','what','is','cost','price','rate','money','Please!', 'tell','give','output','value' ,'','How','much','How','Much'}

li = [ele for ele in li if ele not in articles]
#st.text(li)
item=""
for x in li:
  item=item+" "+ x
#to remove leading and trailing spaces
item=item.strip()
st.text(item)
info_text=info_text.replace('\n','  ')
info_text=info_text.replace('-','')
info_text=info_text.lower()
info_text=info_text.strip()
st.text(info_text)

import re
w11=[]
#For searching the exact item
r4=re.compile(r'(\s*)' + item + r'(\s*[\£\$]?\s*[\.|\:]?\s*((\d+\s?\.\s?\d+)|(\d+))\s*)')  
for x in r4.finditer(info_text):       
        st=x.span()[0]
        en=x.span()[1]
        w11.append(info_text[st:en])
#st.markdown(w11)
#w11=np.array(w11)
#st.dataframe(w11)


