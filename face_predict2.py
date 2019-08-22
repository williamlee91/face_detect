# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:31:07 2018

@author: yuwangwang
"""

import cv2
import sys
from face_train import Model_train
from load_face_dataset import read_name_list

from utils import mtcnn
#from keras import backend as K

# 去除警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Camera_reader():
    
    # 在初始化camera的时候建立模型，并加载已经训练好的模型
    def __init__(self):
        self.model = Model_train()
        self.model.load_model(file_path='E:/sign_system/model/squeezenet1.model.h5')

    def build_camera(self, camera_id, path_name):
        
        print("Loading Deep Face Detector(MTCNN) recognition model ............")
        threshold = [0.6, 0.6, 0.7]
        FaceDetector = mtcnn()
        
        print("Deep Face Detection.............................................")
    
        # 读取dataset数据集下的子文件夹名称
        name_list = read_name_list(path_name)
        
        # 打开摄像头并开始读取画面
        # 捕获指定摄像头的实时视频流
        cap = cv2.VideoCapture(camera_id)
        success, frame = cap.read()

        while True:
             success, frame = cap.read()
                                      
             thickness = (frame.shape[0] + frame.shape[1]) // 350

             faceRects = FaceDetector.detectFace(frame, threshold)
             
             draw = frame.copy()
             #faceRects存的是每个矩形框的中心的坐标以及5个特征点坐标
                          
             for faceRect in faceRects:
                 if faceRect is not None:
                     W = -int(faceRect[0]) + int(faceRect[2])
                     H = -int(faceRect[1]) + int(faceRect[3])
                     paddingH = 0.02 * H
                     paddingW = 0.01 * W
                     crop_img = frame[int(faceRect[1]+paddingH):int(faceRect[3]-paddingH), int(faceRect[0]-paddingW):int(faceRect[2]+paddingW)]
                     
                     print("Face Recognition................................................")
                     label,prob = self.model.face_predict(crop_img) 
                     
                     # 如果模型认为概率高于50%则显示为模型中已有的label
                     if prob > 0.7:    
                         show_name = name_list[label]                     
                     else:
                         show_name = 'Stranger'
                     
                     # 显示名字，字体，字号为1，颜色为粉色，字的线宽为2
                     person_tag = "%s: %.2f" %(show_name, prob)
                     text_end = (int(faceRect[0]) + len(person_tag) * 10,int(faceRect[1]) - 20 )
                     text_start = (max(int(faceRect[0]), 10), max(int(faceRect[1]), 10))
                     cv2.rectangle(draw, text_end, text_start, (0, 255, 0), -1, cv2.LINE_AA )
                     cv2.putText(draw, person_tag, text_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)  
                     
                     #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                     if crop_img is None:
                         continue
                     if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                         continue                                                                      

                 # 在人脸区域画一个正方形出来
                 for i in range(thickness):
                     start = (int(faceRect[0]) + i, int(faceRect[1]) + i)
                     end = (int(faceRect[2] - i), int(faceRect[3]) - i)
                     frame = cv2.rectangle(draw, start, end, (0, 255, 0), 1)  
                
                 # 画出landmarks的
                 for i in range(5, 15, 2):
                     cv2.circle(draw, (int(faceRect[i + 0]), int(faceRect[i + 1])), 2, (0, 255, 0))
                             
             cv2.imshow("Camera", draw)

             # 等待10毫秒看是否有按键输入
             k = cv2.waitKey(10)
             # 如果输入q则退出循环
             if k & 0xFF == ord('q'):
                 break        
        
        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()

        return camera_id, path_name

if __name__ == '__main__':
     if len(sys.argv) != 3:
         print("Usage:%s camera_id path_name\r\n" % (sys.argv[0]))
     else:

         camera = Camera_reader()
     
         camera.build_camera(int(sys.argv[1]), sys.argv[2])