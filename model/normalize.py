import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

def display2(image, name):
    dpi = 80
    height, width = image.shape

    #What size does the figure need to be
    figsize = width/float(dpi), height/float(dpi)

    #Create figure with one axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])

    #Hide markings
    ax.axis('off')

    #display
    ax.imshow(image,cmap='gray')
    fig.canvas.manager.set_window_title(name)
    plt.show()

img = cv.imread('lighting3.png', cv.IMREAD_COLOR_RGB)
gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
#cv.imwrite("testimage.png",gray)
#gray3 = cv.medianBlur(gray,3) #3 works good on some

#height, width = gray3.shape
#half_height = height//2

#gray3 = cv.GaussianBlur(gray,(5,5),0)

#top=gray3[:half_height, :]
#bottom=gray3[half_height:,:]

#gray3 = cv.GaussianBlur(gray,(5,5),0)
#
#f, otsu = cv.threshold(gray,70,255,cv.THRESH_BINARY,cv.THRESH_OTSU)
#adapt = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9,6.5)
array = [gray,gray,gray,gray,gray]
for i in range(len(array)):
    array[i] = cv.adaptiveThreshold(array[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,17,6.5)

#display2(array[4],"aa1a")
#
#ret,thresh0 = cv.threshold(gray,90,255,cv.THRESH_BINARY)
#ret,thresh3 = cv.threshold(gray3,127,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C)
##thresh3 = cv.equalizeHist(thresh3)
#thresh3 = cv.bitwise_not(thresh3)

#top_median = np.median(top)
#bottom_median = np.median(bottom)
#print(top_median, bottom_median)

#ret, top_otsu = cv.threshold(top,127,255,cv.THRESH_BINARY,cv.THRESH_OTSU)
#ret, bottom_otsu = cv.threshold(bottom,127,255,cv.THRESH_BINARY,cv.THRESH_OTSU)

erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
erosion_kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))

for i in range(len(array)):
    array[i]= cv.bitwise_not(array[i])
    array[i] = cv.erode(array[i],erosion_kernel,iterations=1)
    array[i] = cv.dilate(array[i],dilate_kernel,iterations=1)

cv.imwrite("normalized.png",array[4],[cv.IMWRITE_PNG_BILEVEL, 1])

horizontal_contours, hierarchy = cv.findContours(array[4], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img=array[4]
plt.subplot(1,2,1),plt.imshow(img,cmap='gray')



iter=0
for cont in horizontal_contours:
    x,y,w,h = cv.boundingRect(cont)
    if(w>500 or h>500) :
        continue
    rect = cv.rectangle(img,(x,y),(x+w,y+h),(255,255,255),5)
    #crop = img[y:y+h,x:x+w]
    #plt.subplot(1,2,1),plt.imshow(copy)
    #plt.subplot(1,2,2),plt.imshow(crop)
    #if cv.imwrite("./output/cont"+str(iter)+".png",crop, [cv.IMWRITE_PNG_BILEVEL, 1]) == True:
    #   print("success"+str(iter))
    #else:
    #    print("fail"+str(iter))
    print(iter)
    iter=iter+1

#cv.imwrite("temp/mk1segmentation.png",img)
#cv.imwrite("fox_box.jpg",img)

plt.subplot(1,2,2),plt.imshow(img,cmap='gray')
plt.show()

#plt.subplot(1,2,1)
#plt.imshow(gray, cmap='gray')
#for i in range(len(array)):
#    plt.subplot(1,5,i+1),plt.imshow(array[i],cmap='gray'),plt.title(9+i*2)

#plt.subplot(1,5,2),plt.imshow(gray7,cmap='gray'),plt.title("grayscale, blur=3")
#plt.subplot(1,5,3),plt.imshow(gray9, 'gray'),plt.title("grayscale, thresh=100")
#plt.subplot(1,5,4),plt.imshow(thresh3, 'gray'),plt.title("grayscale, blur=3, thresh=100")
#plt.subplot(1,5,5),plt.imshow(thresh3, 'gray'),plt.title("grayscale, blur=3, thresh=100")

#contours,hierarchy = cv.findContours(thresh3,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#cv.drawContours(thresh3,contours,-1,(0,255,0))


#plt.imshow(top_otsu,'gray')


