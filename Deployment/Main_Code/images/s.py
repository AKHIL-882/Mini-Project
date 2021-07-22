"""import cv2


img = cv2.imread("2.jpg")

rows, cols, _ = img.shape

print(rows)
print(cols)


cut_image = img[426: 853, 0: 1280]

cv2.imshow(cut_image,mat)
cv2.imshow(img,mat)"""


import numpy as np
import cv2,mat

image = np.full((300, 300, 3), 255).astype(np.uint8)

cv2.putText(image, '2.jpg', (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0])

cv2.imshow('custom window name', image)
cv2.waitKey(0)
