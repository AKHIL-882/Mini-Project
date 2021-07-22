


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






st.header("Facilitating Surety and Accuracy in the Goods Bought By the Visually Impaired & People With Diminished Vision")
st.subheader("Uplaod Image")


import cam



#user_image = st.file_uploader("kindly Uplaod Image")

st.write("Please Wait!! 😎😎 till the model recogize the image")

# progress = st.progress(0)
# for i in range(100):
#     time.sleep(0.1)
#     progress.progress(i+1)


#st.image(user_image)



image = cv2.imread('1.PNG')





# if True:
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.1)
#         progress.progress(i+1)


path = ""

def blur_and_threshold(gray):
    gray = cv2.GaussianBlur(gray,(3,3),2)
    threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
    return threshold

def biggest_contour(contours,min_area):
    biggest = None
    max_area = 0
    biggest_n=0
    approx_contour=None
    for n,i in enumerate(contours):
            area = cv2.contourArea(i)
         
            
            if area > min_area/10:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
                            biggest_n=n
                            approx_contour=approx
                            
                                                   
    return biggest_n,approx_contour


def order_points(pts):
    pts=pts.reshape(4,2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    
    return warped


def transformation(image):
  image=image.copy()  
  height, width, channels = image.shape
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  image_size=gray.size
  
  threshold=blur_and_threshold(gray)
  
  
  
  
  
  edges = cv2.Canny(threshold,50,150,apertureSize = 7)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  simplified_contours = []


  for cnt in contours:
      hull = cv2.convexHull(cnt)
      simplified_contours.append(cv2.approxPolyDP(hull,
                                0.001*cv2.arcLength(hull,True),True))
  simplified_contours = np.array(simplified_contours)
  biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)

  threshold = cv2.drawContours(image, simplified_contours ,biggest_n, (0,255,0), 1)

  dst = 0
  if approx_contour is not None and len(approx_contour)==4:
      approx_contour=np.float32(approx_contour)
      dst=four_point_transform(threshold,approx_contour)
  croppedImage = dst
  return croppedImage




def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img  





def final_image(rotated):
  
  kernel_sharpening = np.array([[0,-1,0], 
                                [-1, 5,-1],
                                [0,-1,0]])
  
  sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
  sharpened=increase_brightness(sharpened,30)  
  return sharpened






blurred_threshold = transformation(image)
cleaned_image = final_image(blurred_threshold)
cv2.imwrite(path + "Final_Image.jpg", cleaned_image)

original = Image.open('Final_Image.jpg')
st.image(original, use_column_width=True)   




##############################################
# Text Detection from Image
##############################################
info_text = pytesseract.image_to_string(Image.open('2.jpg'))
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