import sys
sys.path.append('train')


import time
import loss_functions
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import logging
from enet import *
from erfnet import *

from network import one_hot_to_image, image_to_one_hot, label_to_one_hot



def restore_session(sess,saver,models_path):

  ckpt = 0
  if not os.path.exists(models_path):
    os.mkdir(models_path)
    os.mkdir(models_path + "/train/")
    os.mkdir(models_path + "/val/")
  
  ckpt = tf.train.get_checkpoint_state(models_path)
  if ckpt:
    print 'Restoring from ',ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
  else:
    ckpt = 0

  return ckpt
    

def save_model(saver,sess,models_path,i):

  if not os.path.exists(models_path):
	os.mkdir(models_path)

  saver.save(sess, models_path + '/model.ckpt', global_step=i)
  print ('Model saved at iteration:',i)


def get_last_iteration(ckpt):

  if ckpt:
    return int(ckpt.model_checkpoint_path.split('-')[1])
  else:
    return 1


class TrainManager(object):

 
	def __init__(self,config,reuse):


		self._config = config
		self._reuse = reuse 
		
		with tf.device('/gpu:0'):
			
			#self._input_images =[]
			#for i in range(len(self._config.sensor_names)):
			#self._input_images.append(tf.placeholder("float",shape=[None,config.image_size[0],config.image_size[1],\
			#config.image_size[2]], name="input_image"))
			self._input_images = tf.placeholder("float",shape=[None,config.image_size[0],config.image_size[1],\
			config.image_size[2]], name="input_image")

			#self._seg_network = tf.placeholder("float",shape=[None,config.image_size[0],config.image_size[1],\
			#5], name="input_seg")	

			self._input_data = []
			self._targets_data = []
			for i in range(len(self._config.targets_names)):
				self._targets_data.append(tf.placeholder(tf.float32, shape=[None, self._config.targets_sizes[i]],name="target_"+self._config.targets_names[i]))
				

			for i in range(len(self._config.inputs_names)):
				self._input_data.append(tf.placeholder(tf.float32, shape=[None, self._config.inputs_sizes[i]],name="input_"+self._config.inputs_names[i]))
				


			#self._output_data = tf.placeholder("float",shape=[config.batch_size,config.output_size], name="output_data")
			self._dout = tf.placeholder("float",shape=[len(config.dropout)])
			self._variable_learning = tf.placeholder("float", name="learning")


		
		#config_gpu = tf.ConfigProto()

		self._feedDict = {}
		self._features = {}

		
  		#config_gpu.log_device_placement=True

		self._create_structure = __import__(config.network_name).create_structure


		
		self._loss_function = getattr(loss_functions, config.loss_function ) # The function to call 

	"""
	def build_seg_network(self,enet):

		with tf.name_scope("Seg_Network"):
			self._seg_network,_,_,_ = self._create_structure_seg(tf, self._input_images[0],enet,self._config.image_size,self._dout,self._config)
	"""


	def build_network(self):

		""" Depends on the actual input """

		with tf.name_scope("Network"):

			self._output_network,self._vis_images,self._features,self._weights = self._create_structure(tf, self._input_images ,self._input_data,self._config.image_size,self._dout,self._config)


	def build_seg_network_gt_one_hot_join(self):

		""" Depends on the actual input """

		with tf.name_scope("Network"):
			network_input = label_to_one_hot(self._input_images,self._config.number_of_labels)
			self._output_network,self._vis_images,self._features,self._weights = self._create_structure(tf, network_input, self._input_data,self._config.image_size,self._dout,self._config)
			self._gray = one_hot_to_image (network_input)
			self._gray = tf.expand_dims(self._gray, -1)

	def build_seg_network_gt_one_hot(self):

		""" Depends on the actual input """

		with tf.name_scope("Network"):

			self._output_network,self._vis_images,self._features,self._weights = self._create_structure(tf, image_to_one_hot(self._input_images,self._config.number_of_labels) ,self._input_data,self._config.image_size,self._dout,self._config)

	def build_rgb_seg_network_one_hot(self):

		""" Depends on the actual input """

		with tf.name_scope("Network"):

			network_input =tf.concat([self._input_images[:,:,:,0:3],image_to_one_hot(self._input_images[:,:,:,3:4],self._config.number_of_labels)],3)

			self._output_network,self._vis_images,self._features,self._weights = self._create_structure(tf, network_input ,self._input_data,self._config.image_size,self._dout,self._config)

	def build_rgb_seg_network_enet(self):

		""" Depends on the actual input """

		self._seg_network = ENet_Small(self._input_images[:,:,:,0:3],self._config.number_of_labels,batch_size=self._config.batch_size,reuse=self._reuse,is_training=self._config.train_segmentation)[0]
		with tf.name_scope("Network"):
			#with tf.variable_scope("Network",reuse=self._reuse):
			#print  self._seg_network 
			self._gray = one_hot_to_image(self._seg_network)
			self._gray = tf.expand_dims(self._gray, -1)
			self._sensor_input = tf.concat([self._input_images[:,:,:,0:3],self._gray],3)
			
			self._output_network,self._vis_images,self._features,self._weights \
			= self._create_structure(tf, self._sensor_input ,self._input_data,self._config.image_size,self._dout,self._config)


	def build_rgb_seg_network_enet_one_hot(self):
		""" Depends on the actual input """

		self._seg_network = ENet_Small(self._input_images[:,:,:,0:3],self._config.number_of_labels,batch_size=self._config.batch_size,reuse=self._reuse,is_training=self._config.train_segmentation)[0]
		with tf.name_scope("Network"):
			#with tf.variable_scope("Network",reuse=self._reuse):
			#print  self._seg_network 

			self._sensor_input = tf.concat([self._input_images[:,:,:,0:3],self._seg_network],3)
			#Just for visualization
			self._gray = one_hot_to_image(self._seg_network)
			self._gray = tf.expand_dims(self._gray, -1)
			
			self._output_network,self._vis_images,self._features,self._weights \
			= self._create_structure(tf, self._sensor_input ,self._input_data,self._config.image_size,self._dout,self._config)


	def build_seg_network_enet_one_hot(self):
		""" Depends on the actual input """

		self._seg_network = ENet_Small(self._input_images[:,:,:,0:3],self._config.number_of_labels,batch_size=self._config.batch_size,reuse=self._reuse,is_training=self._config.train_segmentation)[0]
		with tf.name_scope("Network"):
			#with tf.variable_scope("Network",reuse=self._reuse):
			#print  self._seg_network 

			self._sensor_input = self._seg_network
			#Just for visualization
			self._gray = one_hot_to_image(self._seg_network)
			self._gray = tf.expand_dims(self._gray, -1)
			
			self._output_network,self._vis_images,self._features,self._weights \
			= self._create_structure(tf, self._sensor_input ,self._input_data,self._config.image_size,self._dout,self._config)


	def build_seg_network_erfnet_one_hot(self):
		""" Depends on the actual input """

		self._seg_network = ErfNet_Small(self._input_images[:,:,:,0:3],self._config.number_of_labels,batch_size=self._config.batch_size,reuse=self._reuse,is_training=self._config.train_segmentation)[0]
		with tf.name_scope("Network"):
			#with tf.variable_scope("Network",reuse=self._reuse):
			#print  self._seg_network 

			self._sensor_input = self._seg_network
			#Just for visualization
			self._gray = one_hot_to_image(self._seg_network)
			self._gray = tf.expand_dims(self._gray, -1)
			
			self._output_network,self._vis_images,self._features,self._weights \
			= self._create_structure(tf, self._sensor_input ,self._input_data,self._config.image_size,self._dout,self._config)


	def build_loss(self):
		
		with tf.name_scope("Loss"):
			if hasattr(self._config, 'intermediate_loss'):
				self._loss,self._variable_error,self._variable_energy,self._image_loss,self._branch \
				= self._loss_function(self._output_network,self._seg_network,self._targets_data,self._input_images[:,:,:,3:4],self._input_data[self._config.inputs_names.index("Control")],self._config)
			else:
				self._loss,self._variable_error,self._variable_energy,self._image_loss,self._branch \
				= self._loss_function(self._output_network,self._targets_data,self._input_data[self._config.inputs_names.index("Control")],self._config)
	

	def build_optimization(self):

		""" List of Interesting Parameters """
		#		beta1=0.7,beta2=0.85
		#		beta1=0.99,beta2=0.999
		with tf.name_scope("Optimization"):

			if hasattr(self._config,'segmentation_model_name'):
				train_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - set(slim.get_variables(scope=str(self._config.segmentation_model_name))))
			#print (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
			#train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Network")

			if hasattr(self._config, 'finetune_segmentation') or not(hasattr(self._config,'segmentation_model_name')):
				self._train_step = tf.train.AdamOptimizer(self._variable_learning).minimize(self._loss)
				print("Optimizer: All variables")
			else:
				self._train_step = tf.train.AdamOptimizer(self._variable_learning).minimize(self._loss,var_list=train_vars)
				print("Optimizer: Exclude variables from: ", str(self._config.segmentation_model_name))


	def run_train_step(self,batch_tensor,sess,i):



		capture_time = time.time()
		batch = sess.run(batch_tensor)

		# Get the change in the learning rate]
		decrease_factor = 1
		for position in self._config.training_schedule:
			if i > position[0]:  # already got to this iteration
				decrease_factor =position[1]
				break


		self._feedDict = {self._input_images:batch[0]}

		count=1
		
		for i in range(len(self._config.targets_names)):

			self._feedDict.update({self._targets_data[i]:batch[count]})
			count+= 1



		for i in range(len(self._config.inputs_names)):

			self._feedDict.update({self._input_data[i]:batch[count]})	
			count+= 1


		self._feedDict.update({self._variable_learning: decrease_factor* self._config.learning_rate,self._dout:self._config.dropout})

		sess.run(self._train_step, feed_dict=self._feedDict)


		return time.time() - capture_time


	
	def get_train_step(self):
		return self._train_step

	def get_control_gain(self):
		return self._control_gain

	def get_prob_gain(self):
		return self._prob_gain

	def get_variable_energy(self):
		return self._variable_energy
	
	def get_branch(self):
		return self._branch


	def get_loss(self):

		return self._loss
	
	def get_variable_error(self):
		return self._variable_error

	def get_feed_dict(self):
		return self._feedDict

	def get_network_output(self):
		if self._config.branched_output:
			self._output_network = tf.concat(1,[self._output_network1,self._output_network2])
			
		return self._output_network

	def get_features(self):
		return self._features

	def get_weights(self):
		return self._weights
	
	
	


