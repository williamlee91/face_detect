#coding:utf-8
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, data_size, batch_size, model_path):
        
        self.data_size = data_size
        self.batch_size = batch_size
        self.path = model_path
        
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
            self.landmark_pred = self.sess.graph.get_tensor_by_name("landmark_pred:0")
            print(self.landmark_pred)
                                        
    #rnet and onet minibatch(test)
    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        batch_size = self.batch_size
        minibatch = []
        cur = 0
        #num of all_data
        n = databatch.shape[0]
        while cur < n:
            #split mini-batch
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        #every batch prediction result
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch 
            if m < batch_size:
                keep_inds = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            #cls_prob batch*2
            #bbox_pred batch*4
            cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.inputs: data})
            #num_batch * batch_size *2
            cls_prob_list.append(cls_prob[:real_size])
            #num_batch * batch_size *4
            bbox_pred_list.append(bbox_pred[:real_size])
            #num_batch * batch_size*10
            landmark_pred_list.append(landmark_pred[:real_size])
            #num_of_data*2,num_of_data*4,num_of_data*10
        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
