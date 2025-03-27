import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from hilbert import decode

# Load the image in grayscale
train_names = ['eng_AF_050']
ext = '.jpg'
img = cv.imread('/home/hbarot/Documents/FITPROJECT/train_data/train/' + train_names[0] + ext, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
height, width = img.shape

# Choose a block size that is a power of 2 (e.g., 128)
block_size = 2**7

# Crop the image so that both dimensions are multiples of block_size
new_height = (height // block_size) * block_size
new_width = (width // block_size) * block_size
cropped_img = img[:new_height, :new_width]

# Modify hilbert_flatten to accept the Hilbert order 'p'
def hilbert_flatten(array, p):
    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    # The .T is used for advanced indexing in the original flattening
    L = decode(S, D, p).T.tolist()
    return array[tuple(L)]

def hilbert_unflatten(flat_array, shape, p):
    H, W = shape
    assert H == W, "Image must be square for Hilbert curve reconstruction"
    N = H  # since image is square
    D = 2  # 2D image
    total_points = N * N
    # Do not transpose here so that each coordinate is (x, y)
    coords = decode(np.arange(total_points), D, p).astype(int)
    reconstructed = np.zeros((N, N), dtype=flat_array.dtype)
    for i, (x, y) in enumerate(coords):
        reconstructed[x, y] = flat_array[i]
    return reconstructed

def process_block(block):
    # Determine Hilbert order p for the block (block is square)
    p_block = int(np.log2(block.shape[0]))
    
    # Hilbert flatten the block
    flattened = hilbert_flatten(block, p_block).astype(np.int32)
    
    # Process the flattened array: take differences along the Hilbert curve
    for i in range(1, np.size(flattened) - 2):
        flattened[i] = abs(flattened[i+1] - flattened[i])
    
    # Unflatten back to 2D
    unflattened = hilbert_unflatten(flattened, block.shape, p_block)
    
    # Invert colors
    unflattened = cv.bitwise_not(unflattened)
    
    # Convert to uint8 (required by threshold)
    unflattened = unflattened.astype(np.uint8)
    blurred = cv.GaussianBlur(unflattened, (7,7), 0)
    # Increase contrast via thresholding (using Otsu's method)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Dilation: use a small kernel to make the lines bolder if needed
    kernel = np.ones((1,1), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=1)
    
    # Apply Gaussian blur (adjust kernel size as needed)
    
    
    return dilated#blurred

# Create an empty image for the final result (same size as cropped_img)
final_img = np.zeros_like(cropped_img)

# Loop over the image in non-overlapping blocks, process each block, and place it in final_img
for i in range(0, new_height, block_size):
    for j in range(0, new_width, block_size):
        block = cropped_img[i:i+block_size, j:j+block_size]
        processed_block = process_block(block)
        final_img[i:i+block_size, j:j+block_size] = processed_block

# Save the final stitched image
cv.imwrite('final_processed.jpg', final_img, [cv.IMWRITE_JPEG_QUALITY, 90])
