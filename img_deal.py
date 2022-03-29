import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)

img = cv2.imread('4.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('原始图', imgray)
# print(np.array(imgray))
ret, img = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('二值化', img)
img = cv2.bitwise_not(img)
# cv2.imshow('反转颜色', img)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.dilate(img, kernel=kernel)
# cv2.imshow("dilate_demo", dst)


 #轮廓
contours,hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("number of contours:%d" % len(contours))
cv2.drawContours(dst, contours, -1, (0, 255, 255), 2)

#找到最大区域并填充
area = []
for i in range(len(contours)):
    area.append(cv2.contourArea(contours[i]))
max_idx = np.argmax(area)
for i in range(len(contours)):
    if i != max_idx:
        cv2.fillConvexPoly(dst, contours[i], 0)
# cv2.imshow('去除小连通域', img)


res = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
cv2.imshow('图片尺寸修改', res)
cv2.imwrite('final_4.bmp', res)
print(np.array(res).shape)


cv2.waitKey(10000)

