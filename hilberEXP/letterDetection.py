import cv2
import numpy as np

# Load the image
img = cv2.imread('/home/hbarot/Documents/FITPROJECT/hilberEXP/filtered.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise IOError("Could not open or find the image 'filtered.png'.")



# Find contours in the thresholded image
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find external contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around each detected contour
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    print(x,y,w,h)
    # Filter out very small contours
    #if w < 1 or h < 1:
     #   continue
    # Draw a green rectangle
    cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the result in a window
cv2.imshow("Detected Letters", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image with bounding boxes
cv2.imwrite("detected_letters.jpg", thresh)
print("Image with bounding boxes saved as 'detected_letters.png'")
