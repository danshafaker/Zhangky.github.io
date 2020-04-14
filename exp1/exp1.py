"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
###
#please coding here for solving Task [I].

# 彩色图像直方图

img = cv2.imread('D:/Download/timg2.jpg') #进行直方图均衡化

cv2.imshow('img', img) #显示均衡化的图
img[:, :, 1] =ndimage.gaussian_filter(img[:, :, 1], 3)#高斯平滑
green_img=img[:,:,1]
cv2.imshow('before',green_img)
green_img =ndimage.gaussian_filter(green_img, 30)#高斯平滑
cv2.imshow('after',green_img)
plt.hist(img[:, :, 1].ravel(), bins=20, color='r') #画出均衡化后的直方图
plt.show()
cv2.waitkey();




###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""
'''
###
#please coding here for solving Task[II]
from scipy import ndimage
im = cv2.imread('D:/Download/timg2.jpg')

index = 141
plt.subplot(index)#画图
plt.imshow(im)

for sigma in (2, 5, 10):#循环改变sigma
    im_blur = np.zeros(im.shape, dtype=np.uint8)#建立显示数组
    for i in range(3):  # 对图像的每一个通道都应用高斯滤波
        im_blur[:, :, i] =ndimage.gaussian_filter(im[:, :, i], sigma)#高斯平滑

    index += 1
    plt.subplot(index)#依次创建三个窗口
    plt.imshow(im_blur)#依次显示三张图

plt.show()


'''




"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]


'''
mean = (1, 2)
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, (2, 2), 'raise')   # 2x2x2
plt.figure(figsize=(15, 5))#创立窗口
aix = plt.subplot(141)
aix.hist(x, bins=50, color='r')#显示图像
aix.show()
'''
'''
mean = (1, 1)
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, (800, 800), 'raise')

plt.hist(x.ravel(), bins=128, color='r')
plt.show()'''