from cv2 import *
import streamlit as st
import cv2



videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg",frame)
    st.image(frame)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()

