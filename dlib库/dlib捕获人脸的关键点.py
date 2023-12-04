# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:25:29 2023

@author: 35308
"""

import numpy as np 
import cv2 
import dlib 
#读取图像
img = cv2.imread("y.jpg")
#step1  构造人脸检测器 
detector = dlib.get_frontal_face_detector()
#step2  检测人脸框  使用人脸检测器返回检测到的人脸框
faces = detector(img,0)
#step3  载入模型  加载预测器  68个关键点的模型文件  
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#step4 获取每一张脸的关键点
for face in faces:
    #获取关键点
    shape = predictor(img,face)
    #将关键点转换维坐标(x,y)的形式
    landmarks = np.matrix([[p.x,p.y] for p in shape.parts()])
    #step 5 绘制每一张脸的关键点 绘制shape的每个点
    for idx,point in enumerate(landmarks):
        #当前关键的坐标
        pos = (point[0,0],point[0,1])
        #针对当前关键点，绘制一个实心圆
        cv2.circle(img,pos,2,color=(0,255,0),thickness=-1)
        #字体
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        #利用cv2.putText输出1-68，索引序号加1 显示时从1开始
        cv2.putText(img, str(idx + 1), pos, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
# 绘制结果
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()












































































