import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
train_names=['eng_AF_001']
ext='.jpg'
#test_names=['eng_AF_001']
img = cv.imread('../train_data/train/'+train_names[0]+ext, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
 
equ = cv.equalizeHist(img)
#res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('./train/'+'hist'+train_names[0]+ext,res)