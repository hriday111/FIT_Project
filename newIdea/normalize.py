import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
output_dir = '/home/hbarot/Documents/FITPROJECT/newIdea/cropped_letters/'
os.makedirs(output_dir, exist_ok=True)
for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
    os.makedirs(os.path.join(output_dir, ch), exist_ok=True)
letter_idx = 0
for i in range(1, 7):
    img = cv.imread('/home/hbarot/Documents/FITPROJECT/newIdea/1.png', cv.IMREAD_COLOR_RGB)
    

    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    gray3 = cv.medianBlur(gray,3) #3 works good on some

    height, width = gray3.shape
    half_height = height//2

    gray3 = cv.GaussianBlur(gray,(5,5),0)

    top=gray3[:half_height, :]
    bottom=gray3[half_height:,:]

    #gray3 = cv.GaussianBlur(gray,(5,5),0)
    #
    f, otsu = cv.threshold(gray,70,255,cv.THRESH_BINARY,cv.THRESH_OTSU)
    adapt = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9,6.5)
    array = [gray,gray,gray,gray,gray]
    for i in range(len(array)):
        array[i] = cv.adaptiveThreshold(array[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9+i*2,6.5)
    #
    #
    ret,thresh0 = cv.threshold(gray,90,255,cv.THRESH_BINARY)
    ret,thresh3 = cv.threshold(gray3,127,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    ##thresh3 = cv.equalizeHist(thresh3)
    #thresh3 = cv.bitwise_not(thresh3)

    top_median = np.median(top)
    bottom_median = np.median(bottom)
    print(top_median, bottom_median)

    ret, top_otsu = cv.threshold(top,127,255,cv.THRESH_BINARY,cv.THRESH_OTSU)
    ret, bottom_otsu = cv.threshold(bottom,127,255,cv.THRESH_BINARY,cv.THRESH_OTSU)

    erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))


    for i in range(len(array)):
        array[i]= cv.bitwise_not(array[i])
        array[i] = cv.erode(array[i],erosion_kernel,iterations=1)
        array[i] = cv.dilate(array[i],dilate_kernel,iterations=1)

    horizontal_contours, hierarchy = cv.findContours(array[4], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    img=array[4]
    plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
    
    for cont in horizontal_contours:
        x, y, w, h = cv.boundingRect(cont)
        
        # Padding around the letter (in pixels)
        pad = 5

        # Ensure coordinates stay within image bounds
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img.shape[1])
        y2 = min(y + h + pad, img.shape[0])

        # Crop the padded region
        cropped_letter = img[y1:y2, x1:x2]

        # Add black border if the contour touches image edge
        if x - pad < 0 or y - pad < 0 or x + w + pad > img.shape[1] or y + h + pad > img.shape[0]:
            cropped_letter = cv.copyMakeBorder(
                cropped_letter,
                top=pad if y - pad < 0 else 0,
                bottom=pad if y + h + pad > img.shape[0] else 0,
                left=pad if x - pad < 0 else 0,
                right=pad if x + w + pad > img.shape[1] else 0,
                borderType=cv.BORDER_CONSTANT,
                value=(0, 0, 0)  # Black padding
            )
        
        
        # Save the cropped letter image
        cv.imshow("Cropped Letter", cropped_letter)
        cv.waitKey(1)  # Needed to render the window before input

        # Ask the user for a name
        name = input("Enter Letter:").strip()

        if name:
            letter_path = os.path.join(output_dir, name+"/"+f"{letter_idx}.jpg")
            try:
                cv.imwrite(letter_path, cropped_letter)
                print(f"Saved: {letter_path}")
            except:
                print(f"Could not save {letter_idx}.jpg")
        else:
            print("Skipped.")
        letter_idx+=1
    plt.subplot(1,2,2),plt.imshow(img,cmap='gray')




    plt.show()
    print("FILE ",i," DONE at IDX: ",letter_idx)