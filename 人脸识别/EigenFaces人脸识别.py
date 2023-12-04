# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:50:27 2023

@author: 35308
"""

'''
cv2.face.EigenFaceRecognizer_create ([,num_components[,threeshold]])
使用PCA方法时，保留的分量个数  根据输入数据来具体确定
threshold   进行人脸识别时采用的阈值
'''
import cv2
import numpy as np
# 读取训练图像
images=[]
images.append(cv2.imread("e01.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("e02.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("e11.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("e12.png",cv2.IMREAD_GRAYSCALE))
# 给训练图像贴标签
labels=[0,0,1,1]
# 读取待识别图像
predict_image=cv2.imread("eTest.png",cv2.IMREAD_GRAYSCALE)
# 识别
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.train(images, np.array(labels))  
label,confidence= recognizer.predict(predict_image) 
# 打印识别结果
print("识别标签label=",label)
print("置信度confidence=",confidence)
# 可视化输出
name=["first","second"]  
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(predict_image,name[label],(0,30), font, 0.8,(255,255,255),2)
cv2.imshow("result",predict_image)
cv2.waitKey()
cv2.destroyAllWindows()