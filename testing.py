


import numpy as np
import cv2 as cv

 
img = cv.imread('DB/Train/Squares/1.0_a.png')
dsize = (200, 50)
output = cv.resize(img, dsize)
img_gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray, (5,5), 0)
#Contornos
canny = cv.Canny(img_blur, 50, 150)
contornos,_ = cv.findContours(canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )

#Promedio de los contornos
id=0
print(len(contornos))
areas=[]
for c in contornos:
    epsilon = 0.02*cv.arcLength(c,True)
    approx = cv.approxPolyDP(c,epsilon,True)
    area = cv.contourArea(c)
    areas.append(area)
    print(id,":",area)

print(np.sum(areas)/len(areas))

