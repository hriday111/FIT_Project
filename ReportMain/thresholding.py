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
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray)

img = cv.medianBlur(gray, 3)
# Apply Otsu's Thresholding
_, otsu_thresh = cv.threshold(img, 0, max_val, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Apply Adaptive Mean Thresholding
adaptive_mean = cv.adaptiveThreshold(img, max_val, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, 17, 6.5)

# Apply Adaptive Gaussian Thresholding
adaptive_gaussian = cv.adaptiveThreshold(img, max_val, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY, 17, 5)

# Display results side by side

cv.imwrite(os.path.join(output_dir, 'otsu_thresh.jpg'), otsu_thresh)
cv.imwrite(os.path.join(output_dir, 'adaptive_mean.jpg'), adaptive_mean)
cv.imwrite(os.path.join(output_dir, 'adaptive_gaussian.jpg'), adaptive_gaussian)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu's Thresholding")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(adaptive_mean, cmap='gray')
plt.title("Adaptive Mean Thresholding")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(adaptive_gaussian, cmap='gray')
plt.title("Adaptive Gaussian Thresholding")
plt.axis('off')

plt.tight_layout()
plt.show()