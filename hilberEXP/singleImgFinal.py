import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from hilbert import decode

train_names = ['eng_AF_002']
ext = '.jpg'

img = cv.imread('/home/hbarot/Documents/FITPROJECT/train_data/train/' + train_names[0] + ext, cv.IMREAD_GRAYSCALE)
#convert the img to hsv type
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
assert img is not None, "file could not be read, check with os.path.exists()"
height, width = img.shape

# Crop to the largest square and resize to the nearest power of 2
square_dim = min(height, width)
cropped_img = img[:square_dim, :square_dim]
powers_of_2 = [2**i for i in range(1, 12)]
N = max([p for p in powers_of_2 if p <= square_dim])
p = int(np.log2(N))  # Ensure the Hilbert order matches the image size

# Resize the image
resized_img = np.array(cv.resize(cropped_img, (N, N), interpolation=cv.INTER_AREA))

def hilbert_flatten(array):
    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    # Use the same p (here using 8 previously, but now we use the calculated one)
    L = decode(S, D, p).T.tolist()
    return array[tuple(L)]

hilbert_flattened_img = hilbert_flatten(resized_img).astype(np.int32)
print(hilbert_flattened_img)

def hilbert_unflatten(flat_array, shape, p):
    H, W = shape
    assert H == W, "Image must be square for Hilbert curve reconstruction"
    N = H  # since image is square
    D = 2  # 2D image

    # Total number of pixels
    total_points = N * N

    # Get 2D coordinates without transposing so that each element is a pair (x, y)
    coords = decode(np.arange(total_points), D, p).astype(int)

    # Create an empty image
    reconstructed = np.zeros((N, N), dtype=flat_array.dtype)

    # Map each flattened value back to its (x, y) coordinate
    for i, (x, y) in enumerate(coords):
        reconstructed[x, y] = flat_array[i]
    
    return reconstructed

# Process the flattened image (example processing)
for i in range(1, np.size(hilbert_flattened_img) - 2):
    hilbert_flattened_img[i] = abs(hilbert_flattened_img[i + 1] - hilbert_flattened_img[i])

# Now use the same p when unflattening
hilbert_unflattened_img = hilbert_unflatten(hilbert_flattened_img, resized_img.shape, p)
#invert the colours in the hilber_unflattened_img
hilbert_unflattened_img = cv.bitwise_not(hilbert_unflattened_img)
#increase the contrast
hilbert_unflattened_img = hilbert_unflattened_img.astype(np.uint16)
_, thresh = cv.threshold(hilbert_unflattened_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#cv.imshow('Thresholded Image', thresh)
#cv.waitKey(0)
#thresh_inverted = 255 - thresh
#cv.imshow('Thresholded Image', thresh_inverted)
#cv.waitKey(0)
#to make lines thicker  cv.threshold(hilbert_unflattened_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernel = np.ones((1,1), np.uint16)  # A 2×2 kernel, adjust size as needed
hilbert_unflattened_img = cv.dilate(thresh, kernel, iterations=1)
gaussian_blur15 = cv.GaussianBlur(hilbert_unflattened_img, (15,15), 0)
#7 7 works for now
#hilbert_unflattened_img=cv.morphologyEx(hilbert_unflattened_img, cv.MORPH_CLOSE, kernel)
#save it using cv2
#cv.imwrite('./train/'+'hilbert'+train_names[0]+ext,hilbert_unflattened_img)
cv.imwrite('15GRAY.jpg', gaussian_blur15, [cv.IMWRITE_JPEG_QUALITY, 90])
