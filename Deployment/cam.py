import cv2
import streamlit as st

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 1

while True:
    ret, frame = cam.read()
    if not ret:
        # st.write("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        # st.write("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = format(img_counter)+".png"
        cv2.imwrite(img_name, frame)
        # st.write("written!".format(img_name))
        img_counter += 1
        cam.release()
        cv2.destroyAllWindows()

cam.release()

cv2.destroyAllWindows()