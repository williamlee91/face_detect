# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:43:14 2018

@author: william
"""

"""============================================================================
   2018.4.21 mtcnn-keras-simple takes place of haar for detecting faces
   2018.6.28 mtcnn-tensorflow takes place of mtcnn-keras-simple
   2018.7.25 mtcnn-tensorflow-improved takes place of mtcnn-tensorflow
   2019.1.15 mtcnn-tensorflow-fast takes place of mtcnn-tensorflow
"""
  
import cv2
import numpy as np
import sys
sys.path.append('E:/sign_system/execute_system/MTCNN_Tensorflow')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
#from face_train2 import Model_train
from face_train import Model_train
from load_face_dataset import read_name_list
#from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Camera_reader(object):
            
    def __init__(self):
        
        self.model = Model_train()
        self.model.load_model(file_path=r'E:\sign_system\model\squeezenet1.model.h5')

    def build_camera(self, camera_id, path_name):
        
        #print("Loading Deep Face Detector(MTCNN) recognition model ............")        
        #test_mode = "onet"
        thresh = [0.9, 0.9, 0.8]
        min_face_size = 81 
        stride = 2
        slide_window = False
        #shuffle = False
        #vis = True
        detectors = [None, None, None]
        prefix = ['../execute_system/MTCNN_Tensorflow_fast/data/MTCNN_model/PNet_landmark/PNet', 
                  '../execute_system/MTCNN_Tensorflow_fast/data/MTCNN_model/RNet_landmark/RNet', 
                  '../execute_system/MTCNN_Tensorflow_fast/data/MTCNN_model/ONet_landmark/ONet']
        
        epoch = [40, 36, 36]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        RNet = Detector(R_Net, 24, 1, model_path[1])
        detectors[1] = RNet
        ONet = Detector(O_Net, 48, 1, model_path[2])
        detectors[2] = ONet        
        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
                
        # read names from dataset
        name_list = read_name_list(path_name)        
        cap = cv2.VideoCapture(camera_id)    
        
        #fps1 = cap.get(cv2.CAP_PROP_FPS)
        #size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #size = 640 x 480
       
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,528)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,384)
   
        while True:            

            t1 = cv2.getTickCount()

            success, frame = cap.read() 
            if success:                                     
                thickness = (frame.shape[0] + frame.shape[1]) // 350
                image = np.array(frame)
                
                boxes_c,landmarks = mtcnn_detector.detect(image)
                #print(landmarks.shape)
                
                t2 = cv2.getTickCount()             
                t = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / t            
                print('fps:',fps)
                for i in range(boxes_c.shape[0]):
                    bbox = boxes_c[i, :4]
                    #score = boxes_c[i, 4]
                    cropbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    
                    W = -int(cropbbox[0]) + int(cropbbox[2])
                    H = -int(cropbbox[1]) + int(cropbbox[3])
                    paddingH = 0.02 * H
                    paddingW = 0.01 * W
                    crop_img = frame[int(cropbbox[1]+paddingH):int(cropbbox[3]-paddingH), int(cropbbox[0]-paddingW):int(cropbbox[2]+paddingW)]
                                                    
                    if crop_img is None:
                        continue
                    if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                        continue
    
                    label,prob = self.model.face_predict(crop_img) 
                         
                    if prob > 0.7:    
                        show_name = name_list[label]                     
                    else:
                        show_name = 'Stranger'
                                                                       
                    person_tag = "%s: %.2f" %(show_name, prob)
                    #text_end = (int(cropbbox[0]) + len(person_tag) * 10,int(cropbbox[1]) - 20 )
                    text_start = (max(int(cropbbox[0]), 10), max(int(cropbbox[1]), 10))
                    #cv2.rectangle(draw, text_end, text_start, (255, 255, 0), -1, cv2.LINE_AA )
                    cv2.putText(frame, person_tag, text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)  
                                                                              
                     # rectangle for face area
                    for i in range(thickness):
                        start = (int(cropbbox[0]) + i, int(cropbbox[1]) + i)
                        end = (int(cropbbox[2] - i), int(cropbbox[3]) - i)
                        frame = cv2.rectangle(frame, start, end, (0, 255, 0), 1)  
                    
                     # display the landmarks
                    for i in range(landmarks.shape[0]):
                        for j in range(len(landmarks[i])//2):
                            cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255)) 
                cv2.imshow("Camera", frame)
             
                k = cv2.waitKey(10)
                if k & 0xFF == ord('q'):
                    break        
            else:
                print ('device not find')
                break
        cap.release()
        cv2.destroyAllWindows()
        #return camera_id, path_name

if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("Usage:%s camera_id path_name\r\n" % (sys.argv[0]))
#     else:
         
         camera = Camera_reader()
         camera_id = 0
         path_name = 'E:/sign_system/face_data_for32'
         cameranames = camera.build_camera(camera_id, path_name)
         #camera.build_camera(int(sys.argv[1]), sys.argv[2])
