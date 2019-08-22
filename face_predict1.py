# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:43:14 2018

@author: yuwangwang
"""

"""============================================================================
   2018.2.17版本一，做单目标判断
   1、对输入的摄像头图像进行预测，判断类别
   
   2018.2.21版本二，做多目标判断
   1、对输入的摄像头图像进行判断，显示多目标的名称
"""
  
import cv2
import sys
#import time
from collections import Counter
from face_train import Model_train
from load_face_dataset import read_name_list


# 去除警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Camera_reader():
    
    # 在初始化camera的时候建立模型，并加载已经训练好的模型
    def __init__(self):
        self.model = Model_train()
        self.model.load_model(file_path='E:/sign_system/model/squeezenet1.model.h5')


    def build_camera(self, camera_id, path_name):
        # opencv文件中人脸级联文件的位置，用于帮助识别图像或者视频流中的人脸
        
        
        cascade = cv2.CascadeClassifier('E:/sign_system/opencv/haarcascades/haarcascade_frontalface_alt2.xml')
        # 读取dataset数据集下的子文件夹名称
        name_list = read_name_list(path_name)
        
        # 框住人脸的矩形边框颜色,颜色为绿色      
        color = (0, 255, 0)
        
        # 打开摄像头并开始读取画面
        # 捕获指定摄像头的实时视频流
        cap = cv2.VideoCapture(camera_id)
        #success, frame = cap.read()
        count = 0
        show_name = '' 
        prob = 0
        
        while True:
             
             #t1 = time.time()            
             #检测frame中出现的人脸
             success, frame = cap.read()
             count = count +1
             # 图像灰化             
             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              
             # 利用分类器识别出哪个区域为人脸
             # 参数1：image--待检测图片，一般为灰度图像加快检测速度；
             # 参数2：objects--被检测物体的矩形框向量组；
             # 参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
             # 参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
             # 如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
             # 如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
             # 这种设定值一般用在用户自定义对检测结果的组合程序上；
             # 参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
             # CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
             # 因此这些区域通常不会是人脸所在区域；
             # 参数6、7：minSize和maxSize用来限制得到的目标区域的范围。             
             #faceRects = cascade.detectMultiScale(frame_gray , 1.2, 3) # 识别人脸             
             faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (16, 16))   
             
#             t2 = time.time()
#             time_det = t2 - t1
#             fps = 1 / time_det
#             print("检测时间：%s" %time_det)
#             print("fps:%s" %fps)
             
             #识别检测到的人脸
             for (x, y, w, h) in faceRects:
                 image = frame[y - 10: y + h + 10, x - 10: x + w + 10]

                 #输出所有标签名称
                 #print('name', name_list)
                                  
                 # 利用模型对cv2识别出的人脸进行比对
#                 t4 = time.time()
                 
                 label,prob = self.model.face_predict(image) 
#                 t5 = time.time()
                 
#                 time_one_reg = t5 - t4
#                 print ("单个识别时间： %s" %time_one_reg)
                 
                 # 如果模型认为概率高于50%则显示为模型中已有的label
                 if prob > 0.5:    
                     show_name = name_list[label]
                     #print('id_num:', show_name)
                     
                     #把所有显示过的名字都循环添加
                     #name_labels = [ ]
                     #for i in range(100000):
                         #dic = {}
                         #dic['id_num'] = show_name
                         #name_labels.append(dic)
                 else:
                     show_name = '0'
                 
                 # 显示名字的坐标，字体，字号为1，颜色为粉色，字的线宽为2
                 cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)  
                 
                 # 在人脸区域画一个正方形出来
                 frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness = 2)  
             cv2.imshow("Camera", frame)
             #把所有显示过的名字都循环添加
             name_labels = [ ]
             for i in range(50):
                 dic = {}
                 dic[show_name] = prob
                 name_labels.append(dic)
#             t3 = time.time()
#             time_reg = t3 - t2
#             print("循环识别时间：%s" %time_reg)
             
             # 等待10毫秒看是否有按键输入
             k = cv2.waitKey(10)
             # 如果输入q则退出循环
             if (k & 0xFF == ord('q') or count ==80):
                 break        
        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()

        return name_labels
    
    #找出列表中次数出现最多次数的元素
    def max_list(self,lt):
        temp = 0
        for i in lt:
            if lt.count(i) >temp:
                max_str = i
                temp = lt.count(i)
        return max_str
    
    #统计每个元素出现的次数
    def count_list(self,lt):
        counts = Counter(lt)
        return counts
    
    #修改列表的形式为单一的拼音元素
    def list_analyse(self,lt):
        list01 = []
        for names in lt:
            #print(names)
                a = list(names.values())
            #print(a)
                list01.append(a)
        #print (list01)
        list02 = []  
        for name01 in list01:
            for name02 in name01:
                list02.append(name02)   
        #print (list02)
        return list02 
    
    #统计相同键值的概率值之和
    def prob_sum(self,lt):
        new_dic = {}
        for i in lt:
            keys = i.keys()
            #print ("key1:%s"%keys)
            keys = list(keys)
            #print ("key2:%s"%keys)
            for key in keys: 
                #print ("key3:%s"%key)
                if (key in new_dic.keys()):
                    new_dic[key] = new_dic[key]+i[key]
                else:
                    new_dic[key] = i[key]
        print("概率和:%s"%new_dic)
        return new_dic
    
    #找出概率和最大的键    
    def large_key(self, dic):
        temp = 0
        first_key = 0 
        for key in dic.keys():
            if (dic[key] > temp ):
                
                first_key = key
                temp = dic[key]
        
        print ("工号：%s"%first_key)
        return first_key
    
if __name__ == '__main__':
     if len(sys.argv) != 3:
         print("Usage:%s camera_id path_name\r\n" % (sys.argv[0]))
     else:
         
         camera = Camera_reader()
         cameranames = camera.build_camera(int(sys.argv[1]), sys.argv[2])
         print(cameranames)
         
         ndic = camera.prob_sum(cameranames)
         
         last_workid = camera.large_key(ndic)
         
#         fix_list = camera.list_analyse(cameranames)
#         print(fix_list)
#         
#         counts = camera.count_list(fix_list)
#         print(counts)
#         
#         max_str = camera.max_list(fix_list)
#         print(max_str)
#        
         
          

