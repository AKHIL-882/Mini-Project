##############################################
# Importing the Libraries
##############################################
import streamlit as st
import nltk
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\AKHIL DUGGIRALA\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
import numpy as np
import cv2
import io
import os
import time
import math
from scipy import ndimage
import re
from gtts import gTTS
from googletrans import Translator
import speech_recognition as sr
import PIL
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


# @st.cache
# def load_image(image_file):
#   img = Image.open(image_file)
#   return img


# image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
# if image_file is not None:
#     file_details = {"FileName":image_file.name,"FileType":image_file.type}
#     st.write(file_details)

#     img = load_image(image_file)
#     st.image(img)


#     with open(os.path.join("temp",image_file.name),"wb") as f: 
#       f.write(image_file.getbuffer())         
#     st.success("Saved File")





# path = 'C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/10.jpg'

# if st.button('See Original Image of Tom'):
# original = Image.open('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/2.jpg')
# st.image(original, use_column_width=True)  


# user_image = st.file_uploader("kindly Uplaod Image")
# user_image = cv2.imread(user_image)
# user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
# st.image(user_image)


original = Image.open('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/2.jpg')
st.image(original, use_column_width=True)   

# user_image = cv2.imread('C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/1.PNG')
# user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
    


user_image = path='C:/Users/AKHIL DUGGIRALA/Desktop/StreamLit/Project/images/2.jpg'

    

##############################################
# Calculating the Angle of Image
##############################################


#code for calculating the angle of image
def angle_calculation(image):
  img_before=image 
  key = cv2.waitKey(0)
  img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
  img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
  lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
  angles = []
  for [[x1, y1, x2, y2]] in lines:
      angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
      angles.append(angle)   
  key = cv2.waitKey(0)
  median_angle = np.median(angles)
  angle=int(angle)
  median_angle=int(median_angle)
  return angle,median_angle



# user_image = st.file_uploader("kindly Uplaod Image")



#input image

img_before= cv2.imread(user_image)
try:
  #return error if image doenot contain the data 
  angle,median_angle=angle_calculation(img_before)
  st.write("angle is",angle)
  st.write("median angle",median_angle)
except:
  #saying the user to turn the image
  st.write("Please turn the image and recapture")




else:
  if(median_angle == 90 or median_angle== -90 ):
    img_before = ndimage.rotate(img_before, 90)
  
  
  try:
    
    newdata=pytesseract.image_to_osd(img_before)
    #newdata=pysseract.image_to_osd(img_before)
    st.write(newdata)
    a=re.search('(?<=Rotate: )\d+', newdata).group(0)
    st.write("rotation")
      
  except:
      st.write("Dip of image is not good Please insert proper image")
  else:  
      if(angle >=-5 and angle <=5):
          #img_before = ndimage.rotate(img_before,90)   
          angle=360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(img_before)).group(0))  
          (h, w) = img_before.shape[:2]   
          center = (w / 2, h / 2)
          # Perform the rotation
          M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
          img_rotated=ndimage.rotate(img_before, float(angle) * -1)
          cv2.imwrite('rotated.jpg', img_rotated)
          path='rotated.jpg'
          info_text = pytesseract.image_to_string(Image.open(path))
          st.write(info_text)          
              
      else:   
        img_rotated = ndimage.rotate(img_before, median_angle)
        cv2.imwrite('rotated.jpg', img_rotated) 
        st.write("after rotation")  
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cropping rotated image
        image=Image.open('rotated.jpg')
        image.load()
        image_data = np.asarray(image)
        image_data_bw = image_data.max(axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
        new_image = Image.fromarray(image_data_new)
        new_image.save('rotated.jpg')
        res=cv2.imread('rotated.jpg')
        angle,median_angle=angle_calculation(res)
        st.write("rotated angle",angle)
        try:
          #after rotation if the image is in different angles other than zero then it shows error
          new=pytesseract.image_to_osd(res)
          a=re.search('(?<=Rotate: )\d+', new).group(0)
          st.write("rotation",a)
          a=int(a)
          #if the rotated image is in inversed manner after rotation then we are rotating again
          if(a== 180):
            img_rotated = ndimage.rotate(res, 180)
            cv2.imwrite('rotated.jpg', img_rotated)        
        except:
          st.write("Please recapture the image properly")
        else:
          path='rotated.jpg'
          info_text = pytesseract.image_to_string(Image.open(path))
          st.write(info_text)





