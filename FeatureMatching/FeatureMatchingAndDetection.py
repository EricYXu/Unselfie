import cv2
import numpy as np
import matplotlib.pyplot as plt

#Loads the two images to feature-match
img1 = cv2.imread("Cathedral_Image1.jpeg")
img2 = cv2.imread("Cathedral_Image2.jpeg")


#Initializes the SIFT detector (which finds key features)
sift = cv2.SIFT_create()

#Finds key points and descriptors using SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# img3_rgb = cv2.cvtColor(img3)

image_path = 'FeatureMatchingResult.png'
cv2.imwrite(image_path, img3)
