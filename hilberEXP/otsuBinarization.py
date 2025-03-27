import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
train_names=['eng_AF_001']
ext='.jpg'
#test_names=['eng_AF_001']
img = cv.imread('../train_data/train/'+train_names[0]+ext, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"


gaussian_blur = cv.GaussianBlur(img, (7,7), 0)
ret_gaussian, th_gaussian = cv.threshold(gaussian_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Apply Median Blur
median_blur = cv.medianBlur(img, 3)
ret_median, th_median = cv.threshold(median_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Apply Bilateral Filter
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)
ret_bilateral, th_bilateral = cv.threshold(bilateral_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Plot the results for comparison
titles = ['Original Image', 'Gaussian Blur + Otsu', 'Median Blur + Otsu', 'Bilateral Filter + Otsu']
images = [img, th_gaussian, th_median, th_bilateral]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()