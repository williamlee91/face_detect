# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 22:41:53 2019

@author: Administrator
"""

from PIL import Image
import sys
import os
import urllib
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
 
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
 
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device
 
 
def get_trt_graph(batch_size=128,workspace_size=1<<30):
  # conver pb to FP32pb
  with gfile.FastGFile(model_name,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print("load .pb")
  trt_graph = trt.create_inference_graph(input_graph_def=graph_def, outputs=[output_name],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode=precision_mode)  # Get optimized graph
  print("create trt model done...")
  with gfile.FastGFile("model_tf_FP32.pb",'wb') as f:
    f.write(trt_graph.SerializeToString())
    print("save TRTFP32.pb")
  return trt_graph
 
 
def get_tf_graph():
  with gfile.FastGFile(model_name,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print("load .pb")
  return graph_def
 
 
if "__main__" in __name__:
  model_name = "mobilenetv2_model_tf.pb"
  input_name = "input_1"
  #output_name = "softmax_out"
  output_name = "Logits/Softmax"
  use_tensorrt = True
  precision_mode = "FP32" #"FP16"
  batch_size = 1
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  img_list = glob.glob("/media/xxxxxxx/*.jpg")
 
  if use_tensorrt:
    print("[INFO] converting pb to FP32pb...")
    graph = get_trt_graph(batch_size)
  else:
    print("[INFO] use pb model")
    graph = get_tf_graph()
 
  sess = tf.Session(config=tf_config)
  tf.import_graph_def(graph, name='')
  tf_input = sess.graph.get_tensor_by_name(input_name + ':0') #or use: tf_input = tf.get_default_graph().get_tensor_by_name(input_name + ':0')
  tf_output = sess.graph.get_tensor_by_name(output_name + ':0')
  #tf_output = sess.graph.get_tensor_by_name('Logits/Softmax:0')
  width = int(tf_input.shape.as_list()[1])
  height = int(tf_input.shape.as_list()[2])
  print("input: size:", tf_input.shape.as_list())
  import time
  t=[]
  for img_path in img_list[:1000]:
    t1 = time.time()
    image = Image.open(img_path)
    image = np.array(image.resize((width, height)))
 
    output = sess.run(tf_output, feed_dict={tf_input: image[None, ...]})
    #print("cost:", time.time()-t1)
    t.append(float(time.time()-t1))
    scores = output[0]
    #print("output shape:", np.shape(scores))
    index = np.argmax(scores)
    #print("index:{}, predict:{}".format(index, scores[index]))
  if use_tensorrt:
    print("use tensorrt, image num: {}, all time(s): {}, avg time(s): {}".format(len(t), np.sum(t), np.mean(t)))
  else:
    print("not use tensorrt, image num: {}, all time(s): {}, avg time(s): {}".format(len(t), np.sum(t), np.mean(t)))
  sess.close()
