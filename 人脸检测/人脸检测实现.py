# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:44:44 2023

@author: 35308
"""

import cv2 
#原始图像处理
image = cv2.imread("manypeople.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#加载分类器
'''
scaleFactor  前后两次相继扫描中搜索窗口的缩放比例
minNeighbors  构成检测目标的相邻矩形的最小个数
flags  
minSize 目标的最小尺寸，小于这个尺寸的目标被忽略
maxSize  目标的最大尺寸，大于这个尺寸的目标将被忽略
objects 返回值   目标对象的矩形框向量组
'''
#  
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#人脸检测
'''
 识别图像中的人脸，返回所有人脸的矩形框向量组
# scaleFactor 为了检测到不同大小的目标，通过scalefactor参数把图像长宽同时按照一定比例（默认1.1）逐步缩小，
# 然后检测，这个参数设置的越大，计算速度越快，但可能会错过了某个大小的人脸。
# minNeighbors 构成检测目标的相邻矩形的最小个数，默认值是3
minSize 目标的最小尺寸，小于这个尺寸的目标被忽略
maxSize  目标的最大尺寸，大于这个尺寸的目标将被忽略
objects 返回值   目标对象的矩形框向量组
'''
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.04,
    minNeighbors=18,
    minSize=(8,8))
#打印输出的实现
print("发现{0}张人脸!".format(len(faces)))
print("其位置分别是：")
print(faces)
#标注人脸及显示
for(x,y,w,h) in faces:
  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) 
cv2.imshow("result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
