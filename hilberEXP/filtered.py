import cv2
import numpy as np
import os

# Path to the input image
input_image_path = '/home/hbarot/Documents/FITPROJECT/hilberEXP/az.jpg' 
output_dir = '/home/hbarot/Documents/FITPROJECT/hilberEXP/cropped_letters'
os.makedirs(output_dir, exist_ok=True)

# Load the image in color (or grayscale if preferred)
img = cv2.imread(input_image_path , cv2.IMREAD_GRAYSCALE)
if img is None:
    raise IOError(f"Cannot open image: {input_image_path}")

# Convert to grayscale

# Apply adaptive thresholding to create a binary image.
# Using THRESH_BINARY_INV makes the letters white (foreground) on black.
'''thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)'''
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Optionally, apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
# Optionally, apply some dilation to connect parts of letters if needed:
kernel = np.ones((1,1), np.uint8)
thresh = cv2.dilate(blurred, kernel, iterations=1)

# Find contours in the binary image.
# The RETR_EXTERNAL flag finds only the outer contours.
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Loop over each contour, crop the region and save it.
'''letter_idx = 0
for cnt in contours:
    # Compute the bounding box for the contour
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Optionally, filter out very small contours that are likely noise
    if w < 10 or h < 10:
        continue

    # Crop the region from the original image
    cropped_letter = img[y:y+h, x:x+w]
    
    # Save the cropped letter image
    letter_path = os.path.join(output_dir, f'letter_{letter_idx}.jpg')
    cv2.imwrite(letter_path, cropped_letter)
    letter_idx += 1

print(f"Detected and saved {letter_idx} letter(s) in '{output_dir}' directory.")'''

# Loop over each contour to draw bounding boxes on the original image.
'''for cnt in contours:
    # Compute the bounding rectangle for each contour.
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filter out small contours that are likely noise.
    if w < 10 or h < 10:
        continue

    # Draw a rectangle around the detected letter on the original image.
    # Here (255, 0, 0) is blue color and thickness is 2.
    cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 2)'''

# Instead of saving the cropped letters, display the original image with bounding boxes.
cv2.imshow("Detected Letters", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()