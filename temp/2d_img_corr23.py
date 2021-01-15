import os
import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from skimage.feature import match_template

IMAGE_1 = "./images/01.png"
IMAGE_2 = "./images/01.png"
"""
def correlate2d(img1, img2):
    corr = 0
    for i in range(img1.shape[0]):
        print(abs(np.corrcoef(img1[i], img2[i])[0, 1]), i + 1, img1.shape[0])
        corr += abs(np.corrcoef(img1[i], img2[i])[0, 1])
    return corr / img1.shape[0]
"""
img1 = cv2.imread(IMAGE_1)
img2 = cv2.imread(IMAGE_2)

gray1 = np.asarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), dtype=np.float32)
gray2 = np.asarray(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), dtype=np.float32)

gray1 -= gray1.mean()
gray2 -= gray2.mean()

#cv2.imshow('image1', gray1)
#cv2.imshow('image2', gray2)

#corr = correlate2d(gray1, gray2)
#print(corr)
#corr = signal.correlate2d(gray1, gray2, mode="same")
corr = match_template(gray1, gray2, pad_input=True)
print(corr)
print(gray1.shape, gray2.shape, corr.shape)
plt.imshow(corr, cmap='hot')
plt.show()
cv2.imshow('corr', corr)

cv2.waitKey(0)
cv2.destroyAllWindows()
