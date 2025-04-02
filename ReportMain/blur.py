import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
output_dir = '/home/hbarot/Documents/FITPROJECT/ReportMain/output'
os.makedirs(output_dir, exist_ok=True)
for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
    os.makedirs(os.path.join(output_dir, ch), exist_ok=True)
letter_idx = 0
img = cv.imread('/home/hbarot/Documents/FITPROJECT/ReportMain/input/L1.png', cv.IMREAD_COLOR_RGB)


gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

gray1 = cv.medianBlur(gray, 3)
gray2 = cv.GaussianBlur(gray, (3, 3), 0)
gray3 = cv.bilateralFilter(gray, 9, 75, 75)
# Display the filtered images side by side
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray1, cmap='gray')
plt.title('Median Blur')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray2, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray3, cmap='gray')
plt.title('Bilateral Filter')
plt.axis('off')

plt.tight_layout()
plt.show()