# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:49:46 2019

@author: Administrator
"""

import os
import sys
import tensorflow as tf
sys.path.append('E:/MTCNN_Tensorflow_improve')
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from tensorflow.python.framework import graph_util


MODEL_DIR = r'E:/MTCNN_Tensorflow_improve/data/MTCNN_model_V2'
OUTPUT_DIR = r'E:/sign_system/execute_system/haar_extract/MTCNN_new'

def save_pnet():
    model_path = os.path.join(MODEL_DIR, "PNet_landmark/PNet-40")
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            #define tensor and op in graph(-1,1)
            image_op = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input')
            cls_prob, bbox_pred, _ = P_Net(image_op, training=False)
            output_node_names = 'cls_prob,bbox_pred,landmark_pred'
            
            saver = tf.train.Saver() 
            saver.restore(sess, model_path)

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
                
            # Freeze the graph def
            output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                input_graph_def = input_graph_def, 
                                                output_node_names=output_node_names.split(",") )
        
            output_pnet = os.path.join(OUTPUT_DIR, 'pnet.pb')
            # Serialize and dump the output graph to the filesystem
            
            with tf.gfile.GFile(output_pnet, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_pnet))
    #for op in graph.get_operations():
        #print(op.name, op.values())


def save_rnet():
    model_path = os.path.join(MODEL_DIR, "RNet_landmark/RNet-36")
    graph = tf.Graph()          
    with graph.as_default():
        with tf.Session() as sess:
            #define tensor and op in graph(-1,1)
            image_op = tf.placeholder(tf.float32, shape=[None, 24, 24, 3], name='input')
            #figure out landmark            
            cls_prob, bbox_pred, landmark_pred = R_Net(image_op, training=False)
            output_node_names = 'cls_prob,bbox_pred,landmark_pred'
            
            saver = tf.train.Saver() 
            saver.restore(sess, model_path)
        
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
                
            # Freeze the graph def
            output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                input_graph_def = input_graph_def, 
                                                output_node_names=output_node_names.split(",") )
        
            output_rnet = os.path.join(OUTPUT_DIR, 'rnet.pb')
            # Serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_rnet, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_rnet))
            
def save_onet():
    model_path = os.path.join(MODEL_DIR, "ONet_landmark/ONet-36")
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            #define tensor and op in graph(-1,1)
            image_op = tf.placeholder(tf.float32, shape=[None, 48, 48, 3], name='input')
            #figure out landmark            
            cls_prob, bbox_pred, landmark_pred = O_Net(image_op, training=False)
            output_node_names = 'cls_prob,bbox_pred,landmark_pred'
            
            saver = tf.train.Saver() 
            saver.restore(sess, model_path)
        
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
                
            # Freeze the graph def
            output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                input_graph_def = input_graph_def, 
                                                output_node_names=output_node_names.split(",") )       
            
            output_onet =os.path.join(OUTPUT_DIR, 'onet.pb')
            # Serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_onet, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_onet))


def main():
    save_pnet()
    save_rnet()
    save_onet()

if __name__ == "__main__":
    main()
        

