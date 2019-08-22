# -*- coding: utf-8 -*-
import tensorflow as tf
tf.reset_default_graph()
class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, model_path):
        
        self.path = model_path
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            
            output_graph_def = tf.GraphDef()
            with open(self.path, "rb") as f:   #读取模型
                output_graph_def.ParseFromString(f.read())  # rb
                _ = tf.import_graph_def(output_graph_def, name="")
                
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options,)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config) 
            init = tf.global_variables_initializer()
            self.sess.run(init)
            
#            op = self.sess.graph.get_operations()
#            for i,m in enumerate(op):
#                print('op{}:'.format(i),m.values())
            
            self.inputs = self.sess.graph.get_tensor_by_name("input:0")  
            print(self.inputs)
            self.cls_prob = self.sess.graph.get_tensor_by_name("cls_prob:0")
            print(self.cls_prob)
            self.bbox_pred = self.sess.graph.get_tensor_by_name("bbox_pred:0")
            print(self.bbox_pred)
            
    def predict(self, databatch):
        height, width, channels = databatch.shape
        databatch = databatch.reshape(1, height, width, channels)
        # print(height, width)
        feed_dict = {self.inputs:databatch}
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict=feed_dict)
        return cls_prob, bbox_pred
