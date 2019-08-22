# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:19:12 2019

@author: Administrator
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import sys
sys.path.append('E:/sign_system/execute_system/MTCNN_Tensorflow_improve')
from Detector.MtcnnDetector import MtcnnDetector
from Detector.detector import Detector
from Detector.fcn_detector import FcnDetector

class Face(object):
    def __init__(self, path):
        self.path = path  
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(self.path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tensors = tf.import_graph_def(output_graph_def, name="")
                print("tensors:",tensors)

            # define sess
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options,)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
 
#            op = self.sess.graph.get_operations()
#            for i,m in enumerate(op):
#                print('op{}:'.format(i),m.values())
 
            self.inputs= self.sess.graph.get_tensor_by_name("input_1:0")  # 具体名称看上一段代码的input.name
            print("input_X:",self.inputs)
            self.out = self.sess.graph.get_tensor_by_name("flatten/Reshape:0")  # 具体名称看上一段代码的output.name
            print("input_X:",self.out)

    def get_feature(self, inputs):
        feed_dict = {self.inputs: inputs}
        feature = self.sess.run(self.out, feed_dict=feed_dict)
        return feature       
                                                
    def crop_pic_extract(self):          
        path = r'E:\sign_system\extract'
        dirs = os.listdir(path)
        print(dirs)
        time_all = 0.0
        for file in dirs:
            labels = file
            if not labels:
                break
            path_pic =  'E:/sign_system/extract'+'/'+labels
            dirspic = os.listdir(path_pic)
            f_all = 0 
            num = 0 
            for i in dirspic:
                
                image = cv2.imread(path_pic+'/'+i)
                if image is None:
                    continue
                # Turn the image into an array.
                image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
                image = image.reshape((1, 96, 96, 3)) 
                image = image.astype('float32')
                image = image - 127.5
                image = image * 0.0078125   
                      
                t1 = time.time()
                
                f1 = self.get_feature(image)
                t2 = time.time()
                time_all = t2 - t1 + time_all
                print('time cost:',t2-t1)
                f_all = f_all + f1
                num = num + 1
            f_all = f_all / num
            f=open('E:/sign_system/execute_system/crop_feature/test.txt','a')
            f.write(labels)
            f.write('\n')
            count = 0
            for element in f_all.flat:
                count = count + 1
                f.write(str(element))
                f.write(' ')
                if int(count%10) ==0:
                    f.write('\n')
            f.write('\n')
            f.close()  
                
    def build_camera(self, camera_id, path):
        count = 10000
        thresh = [0.9, 0.9, 0.8]
        min_face_size = 100
        stride = 2
        slide_window = False
        detectors = [None, None, None]
        path_base = 'E:/sign_system/execute_system/haar_extract'
        paths = [os.path.join(path_base,'pnet.pb'), 
                os.path.join(path_base,'rnet.pb'), 
                os.path.join(path_base,'onet.pb')]
        PNet = FcnDetector(paths[0])
        detectors[0] = PNet
        RNet = Detector(24, 1, paths[1])
        detectors[1] = RNet
        ONet = Detector(48, 1, paths[2])
        detectors[2] = ONet        
        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
        cap = cv2.VideoCapture(camera_id)
        num = 0
        cur = self.read_feature(path)
        while True:
            success, frame = cap.read()
            thickness = (frame.shape[0] + frame.shape[1]) // 350
            print(frame.shape)
            if success:
                t1 = time.time() 
                img = np.array(frame)
                boxes_c,landmarks = mtcnn_detector.detect(img)
                #print(boxes_c)
                for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]
                        #score = boxes_c[i, 4]
                        cropbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        W = -int(cropbbox[0]) + int(cropbbox[2])
                        H = -int(cropbbox[1]) + int(cropbbox[3])
                        paddingH = 0.02 * H
                        paddingW = 0.01 * W
                        crop_img = frame[int(cropbbox[1]+paddingH):int(cropbbox[3]-paddingH), 
                                       int(cropbbox[0]-paddingW):int(cropbbox[2]+paddingW)]
                        
                        if crop_img is None:
                            continue
                        
                        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                            continue
                        
                        image = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_CUBIC)  
                        image = image.reshape((1, 96, 96, 3)) 
                                                 
                        image = image.astype('float32')
                        image = image - 127.5
                        image = image * 0.0078125   
                        f1 = self.get_feature(image)
                        
                        f1 = f1.reshape(256)
      
                        #计算距离
                        d1 = 0
                        show_name = ''
                        for n,v in cur.items():
                            v = np.array(v)
                            d=np.dot(v,f1)/(np.linalg.norm(v)*np.linalg.norm(f1))
                            #print(d)
                            if d > d1:
                                d1 = d
                                show_name = str(n)
                            else:
                                pass
                        #print(show_name)
                        t2 = time.time()
                        delta_t = t2-t1
                        text_start = (max(int(cropbbox[0]), 10), max(int(cropbbox[1]), 10))
                        cv2.putText(frame, show_name, text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)  
                        cv2.putText(frame, "FPS:" + '%.04f'%(1/delta_t),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 255), 2)                                                          
                        # rectangle for face area
                        for i in range(thickness):
                            start = (int(cropbbox[0]) + i, int(cropbbox[1]) + i)
                            end = (int(cropbbox[2] - i), int(cropbbox[3]) - i)
                            frame = cv2.rectangle(frame, start, end, (0, 255, 0), 1)  
                            
                        # display the landmarks
                        for i in range(landmarks.shape[0]):
                            for j in range(len(landmarks[i])//2):
                                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255)) 
                        num = num +1                
                cv2.imshow("Camera", frame)
                k = cv2.waitKey(10)
                # 如果输入q则退出循环
                if (k & 0xFF == ord('q') or count == num):
                    break  
            else:
                print ('device not find')
                break
        cap.release()
        cv2.destroyAllWindows()                     
            
    def read_feature(self,path_name):      
        f = open(path_name,'r')
        cur_entry = {}
        numlist = []
        while True:
            line = f.readline().strip('\n')   
            if not line:
                break
            names = line
            for i in range(26):
                nums = f.readline().strip('\n').split(' ')
                for j in nums:
                    try: 
                        float(j)
                        numlist.append(float(j)) 
                    except ValueError:  
                        pass  
            cur_entry[names] = numlist
            numlist = []
        f.close()
        return cur_entry
    
if __name__ == '__main__':
    path = r'E:/sign_system/execute_system/haar_extract/my_model.pb'   
    hmodel = Face(path)
    
    #hmodel.crop_pic_extract()
    
    camera_id = 0
    path_name = r'E:/sign_system/execute_system/crop_feature/test.txt'
    hmodel.build_camera(camera_id,path_name)   