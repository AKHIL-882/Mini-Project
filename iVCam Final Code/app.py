##############################################
# Importing the required Libraries
##############################################
import streamlit as st
import nltk
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\AKHIL DUGGIRALA\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
from gtts import gTTS
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
import os


from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")



##############################################
# Calling and Openning the Web cam
##############################################

#import cam




##############################################
# Preprocessing the Image
##############################################

path = ""
image = cv2.imread("1.jpg")




# ## **Use Gaussian Blurring combined with Adaptive Threshold** 

# def blur_and_threshold(gray):
#     gray = cv2.GaussianBlur(gray,(3,3),2)
#     threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#     threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
#     return threshold


# # ## **Find the Biggest Contour** 

# # **Note: We made sure the minimum contour is bigger than 1/10 size of the whole picture. This helps in removing very small contours (noise) from our dataset**


# def biggest_contour(contours,min_area):
#     biggest = None
#     max_area = 0
#     biggest_n=0
#     approx_contour=None
#     for n,i in enumerate(contours):
#             area = cv2.contourArea(i)
         
            
#             if area > min_area/10:
#                     peri = cv2.arcLength(i,True)
#                     approx = cv2.approxPolyDP(i,0.02*peri,True)
#                     if area > max_area and len(approx)==4:
#                             biggest = approx
#                             max_area = area
#                             biggest_n=n
#                             approx_contour=approx
                            
                                                   
#     return biggest_n,approx_contour


# def order_points(pts):
#     # initialzie a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     pts=pts.reshape(4,2)
#     rect = np.zeros((4, 2), dtype = "float32")

#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis = 1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     # now, compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis = 1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     # return the ordered coordinates
#     return rect


# # ## Find the exact (x,y) coordinates of the biggest contour and crop it out


# def four_point_transform(image, pts):
#     # obtain a consistent order of the points and unpack them
#     # individually
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect

#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
   

#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))

#     # now that we have the dimensions of the new image, construct
#     # the set of destination points to obtain a "birds eye view",
#     # (i.e. top-down view) of the image, again specifying points
#     # in the top-left, top-right, bottom-right, and bottom-left
#     # order
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype = "float32")

#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

#     # return the warped image
#     return warped


# # # Transformation the image

# # **1. Convert the image to grayscale**

# # **2. Remove noise and smoothen out the image by applying blurring and thresholding techniques**

# # **3. Use Canny Edge Detection to find the edges**

# # **4. Find the biggest contour and crop it out**


# def transformation(image):
#   image=image.copy()  
#   height, width, channels = image.shape
#   gray = np.zeros(image.shape[:-1], dtype=image.dtype)
#   gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#   image_size=gray.size
  
#   threshold=blur_and_threshold(gray)
#   # We need two threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal 
#   # are sure to be edges and those below minVal are sure to be non-edges, so discarded. 
#   #  Those who lie between these two thresholds are classified edges or non-edges based on their connectivity.
#   # If they are connected to "sure-edge" pixels, they are considered to be part of edges. 
#   #  Otherwise, they are also discarded
#   edges = cv2.Canny(threshold,50,150,apertureSize = 7)
#   contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   simplified_contours = []


#   for cnt in contours:
#       hull = cv2.convexHull(cnt)
#       simplified_contours.append(cv2.approxPolyDP(hull,
#                                 0.001*cv2.arcLength(hull,True),True))
#   simplified_contours = np.array(simplified_contours)
#   biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)

#   threshold = cv2.drawContours(image, simplified_contours ,biggest_n, (0,255,0), 1)

#   dst = 0
#   if approx_contour is not None and len(approx_contour)==4:
#       approx_contour=np.float32(approx_contour)
#       dst=four_point_transform(threshold,approx_contour)
#   croppedImage = dst
#   return croppedImage


# # **Increase the brightness of the image by playing with the "V" value (from HSV)**

# def increase_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value
#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img  


# # **Sharpen the image using Kernel Sharpening Technique**


# def final_image(rotated):
#   # Create our shapening kernel, it must equal to one eventually
#   kernel_sharpening = np.array([[0,-1,0], 
#                                 [-1, 5,-1],
#                                 [0,-1,0]])
#   # applying the sharpening kernel to the input image & displaying it.
#   sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
#   sharpened=increase_brightness(sharpened,30)  
#   return sharpened


# # ## 1. Pass the image through the transformation function to crop out the biggest contour

# # ## 2. Brighten & Sharpen the image to get a final cleaned image

# print(image.shape[2])


# blurred_threshold = transformation(image)
# cleaned_image = final_image(blurred_threshold)
# cv2.imwrite(path + "Final_Image_new.jpg", cleaned_image)



##############################################
# Angle and Rotation of Image
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




#input image


img_before= cv2.imread('1.jpg')
try:
  #return error if image doenot contain the data 
  angle,median_angle=angle_calculation(img_before)
 # st.write(" angle is",angle)
  #st.write("median angle",median_angle)
except:
  #saying the user to turn the image
  #st.write("Please turn the image and recapture")
  language='en'
  x="Please turn the image and recapture"
  audio=gTTS(text=x,lang=language,slow=True)

  audio.save("2.mp3")
  os.system("2.mp3")
  # voice - > Audio format


else:
  if(median_angle == 90 or median_angle== -90 ):
    img_before = ndimage.rotate(img_before, 90)
  
  
  try:
    
    newdata=pytesseract.image_to_osd(img_before)
    #newdata=pysseract.image_to_osd(img_before)
    #st.write(newdata)
    a=re.search('(?<=Rotate: )\d+', newdata).group(0)
    #st.write("rotation")
      
  except:
      #st.write("Dip of image is not good Please insert proper image")
      language='en'
      x="Dip of image is not good Please insert proper image"
      final_image_new_text_detected_array = pytesseract.image_to_string(Image.open("1.jpg"))
      #st.write(final_image_new_text_detected_array)
      language='en'
      x="Image processing has been completed."
      audio=gTTS(text=x,lang=language,slow=True)
      audio.save("3.mp3")
      os.system("3.mp3")

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
          final_image_new_text_detected_array = pytesseract.image_to_string(Image.open(path))
         # st.write(final_image_new_text_detected_array)          
              
      else:   
        img_rotated = ndimage.rotate(img_before, median_angle)
        cv2.imwrite('rotated.jpg', img_rotated) 
        #st.write("after rotation")  
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
        #st.write("rotated angle",angle)
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
          language='en'
          x="Please recapture the image properly"
          audio=gTTS(text=x,lang=language,slow=True)

          audio.save("4.mp3")
          os.system("4.mp3")

          # voice - > Audio 
        else:
          path='rotated.jpg'
          final_image_new_text_detected_array = pytesseract.image_to_string(Image.open(path))
          st.write(final_image_new_text_detected_array)






##############################################
# Text Detection
##############################################

# path='Final_Image_new.jpg'
# final_image_new_text_detected = pytesseract.image_to_string(Image.open(path))
# st.write(final_image_new_text_detected)
#st.write(type(final_image_new_text_detected))


##############################################
# Converting into an array of values
##############################################

#final_image_new_text_detected_array = str.split(final_image_new_text_detected_array)
#st.write(final_image_new_text_detected_array)



##############################################
# Speech Recognition
##############################################




##############################################
# Converting to lower case
##############################################
# for i in range(len(final_image_new_text_detected_array)):
#    final_image_new_text_detected_array[i] = final_image_new_text_detected_array[i].lower()
final_image_new_text_detected_array = final_image_new_text_detected_array.lower()
#st.write(final_image_new_text_detected_array)





##############################################
# User Speech as Input
##############################################
st.subheader("Query Section")
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #st.write("Kindly ask you query")
        language='en'
        x="Kindly ask you query"
        audio=gTTS(text=x,lang=language,slow=True)

        audio.save("6.mp3")
        os.system("6.mp3")

        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            #st.write("Your Query : ", text)
        except:
            #st.write("Please Ask your query correctly")
            language='en'
            x="Please Ask your query correctly"
            audio=gTTS(text=x,lang=language,slow=True)

            audio.save("7.mp3")
            os.system("7.mp3")
        return text

##############################################
# Speech Activates when click on button
##############################################
#if st.button("Speech as Input"):
 #   recognized = takecommand()






############################################
# Removing Stopwords
##############################################
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize

# text_tokens = word_tokenize(recognized)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

# #st.write(tokens_without_sw)

# filtered_sentence = (" ").join(tokens_without_sw)

# recognized = filtered_sentence
# st.write(recognized)
# st.write(type(recognized))




# total amount


##############################################
# Converting Speech into token of array values
##############################################
# speech_token = recognized.split()
# st.write(type(speech_token))

# for i in range(len(speech_token)):
#    speech_token[i] = speech_token[i].lower()
# st.write(speech_token)









##############################################
# User Speech Mapping
##############################################


def timer(data):
    p=re.compile(r'\s?\d{2}\s?\:\s?\d{2}\s?')
    time=[]
    for x in p.finditer(data):
        #st.write(x)
        st1=x.span()[0]
        en=x.span()[1]
        time.append(data[st1:en])
    return time
# time=time(final_image_new_text_detected_array)
# st.write(time)

import re
def final_result(item):
  w11 = item +"not found"
  item = item
  #print(item)
  if(item == 'time'):
     time = timer(final_image_new_text_detected_array)
     #st.write(time)
  if(item == 'date'):
    pass

  
  else:
    #For searching the exact item
    r4=re.compile(r'(\s*)' + item + r'(\s*[\£\$]?\s*[\.|\:]?\s*((\d+\s?\.\s?\d+)|(\d+))\s*)')
    for x in r4.finditer(final_image_new_text_detected_array):
            #st.write(x)
            st1=x.span()[0]
            en=x.span()[1]
            w11 = final_image_new_text_detected_array[st1:en]
            #st.write(w11)
  return w11


def related(item):
  item=item
  v = re.compile(r'(\s?[A-Za-z]*\s*)?'+item+r'(\s*[A-Za-z]*\s*[A-Za-z]*[\£\$]?\s*[\.|\:]?\s*((\d+\s?\.\s?\d+)|(\d+)))')
  w11="No related items Found"
  for x in v.finditer(final_image_new_text_detected_array):
    st1=x.span()[0]
    en=x.span()[1]
    w11 = final_image_new_text_detected_array[st1:en]
    #st.write(w11)
  return w11

  # elif (item):
  #   l=len(li)
  #   def search(w11,item):
  #     s=item
  #     s=str(s)
  #     w11 = ""
  #     r4=re.compile(s+ r'(\s*[\£\$]?\s*[\.|\:]?\s*((\d+\s?\.\s?\d+)|(\d+)))')
  #     for x in r4.finditer(final_image_new_text_detected):
  #         #st.write(x)
  #         st1=x.span()[0]
  #         en=x.span()[1]
  #         w11 = final_image_new_text_detected[st1:en]
  #         st.write(w11)
  #     return w11
  #   i=0
  #   j=l
  #   output=[]
  #   w11=[]
  #   while i<j:
  #     output=search(w11,li[i])
  #     i+=1
  #   st.write(output)

#st.write(w11)

#st.button("Click me for no reason")
 
# Create a button, that when clicked, shows a text
c1,c2=st.beta_columns(2)
#c1.button("camera/upload",key="1")
#c2.button("TextToSpeech",key="2")
c3,c4=st.beta_columns(2)
c5,c6=st.beta_columns(2)
#c3.button("search",key="3")
#c4.button("related",key="4")


language="en"
instructions = "Click on top to read our recipt click on middle to search an item and click at bottom to search relavent items"
audio=gTTS(text=instructions,lang=language,slow=True)
audio.save("8.mp3")
os.system("8.mp3")

if c1.button("  ReadOut Recipt Data "):
  language="en"
  audio=gTTS(text=final_image_new_text_detected_array,lang=language,slow=True)
  audio.save("9.mp3")
  os.system("9.mp3")
  st.balloons()
  st.balloons()



if c3.button("   Search an Item    "):
  recognized = takecommand()
  text_tokens = word_tokenize(recognized)
  tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
  filtered_sentence = (" ").join(tokens_without_sw)
  recognized = filtered_sentence
  item = recognized 
  item=item.lower()
  w11 = final_result(item)
  language='en'
  audio=gTTS(text=w11,lang=language,slow=True)
  audio.save("10.mp3")
  os.system("10.mp3")
  st.balloons()
  st.balloons()

  

if c5.button("Search any related item"):
  recognized = takecommand()
  item = recognized
  item=item.lower()
  w11 = related(item)
  language="en"
  audio=gTTS(text=w11,lang=language,slow=True)
  audio.save("11.mp3")
  os.system("11.mp3")
  st.balloons()
  st.balloons()



##############################################
# Copying The Output into a file
##############################################








st.balloons()
st.balloons()

st.balloons()
st.balloons()






