# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:25:53 2023

@author: 35308
"""

'''
cv2.face_FaceRecognizer.train   给定的数据核相关标签训练生成的实例模型
None = cv2.face_FaceRecognizer.train(src,labels)   labels 标签，人脸图像对应的标签
cv2.face_FaceRecognizer.predict()  对一个待识别人脸图像进行判断，寻找与当前图像距离最佳的人脸图像 
cv2.face.LBPHFaceRecognizer_create LBPH识别器实例模型   cv2.face_FaceRecognizer.train完成训练   cv2.face_FaceRecognizer.predict()完成人脸识别
cv2.face.LBPHFaceRecognizer_create ([,radius[,neighbours[,grid_x[,grid_y[,threshold]]]]])
radius半径值   1
neighbors  领域点的个数  默认8  
grid_x  LBP特征图划分维一个个单元时在水平方向上的单元个数
grid_y  LBP特征图划分维一个个单元时在垂直方向上的单元个数
threshold   预测时使用的阈值  
'''
import cv2
import numpy as np
# 读取训练图像
images=[]
images.append(cv2.imread("a1.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("a2.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("b1.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("b2.png",cv2.IMREAD_GRAYSCALE))
# 给训练图像贴标签
labels=[0,0,1,1]
# 读取待识别图像
predict_image=cv2.imread("a3.png",cv2.IMREAD_GRAYSCALE)
# 识别
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))  
label,confidence= recognizer.predict(predict_image) 
# 打印识别结果
print("对应的标签label=",label)
print("置信度confidence=",confidence)