
import sys
import os

import socket
import scipy
import re

import math

from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
import tensorflow as tf
import time

import pygame
sys.path.append('../')
sys.path.append('../train')
from pygame.locals import *
from socket_util import *

from planner import *
from client import *


from codification import *

from training_manager import TrainManager
import machine_output_functions


from drawing_tools import *

def restore_session(sess,saver,models_path):

  ckpt = 0
  if not os.path.exists(models_path):
    os.mkdir( models_path)
  
  ckpt = tf.train.get_checkpoint_state(models_path)
  if ckpt:
    print 'Restoring from ',ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
  else:
    ckpt = 0

  return ckpt


def load_system(config):
  config.batch_size =1
  config.is_training=False

  training_manager= TrainManager(config)


  training_manager.build_network()


  """ Initializing Session as variables that control the session """
  



  return training_manager



class GTAMachine(object):


	def __init__(self,gpu_number,experiment_name,use_planner =False):


		conf_module  = __import__(experiment_name)
		self._config = conf_module.configInput()
		self._use_planner = use_planner
		config_gpu = tf.ConfigProto()
		config_gpu.gpu_options.visible_device_list=gpu_number
		self._sess = tf.Session(config=config_gpu)

		self._straight_button = False
		self._left_button = False
		self._right_button = False
		self._recording= False
		self._mean_image = np.load(self._config.save_data_stats + '/meanimage.npy')
		self._train_manager =  load_system(conf_module.configTrain())
		self._rear = False
		self._current_epi = 0
		self._current_pos =[0,0,0]
		self._current_goal=[100,100]
		self._number_completions =0

		self._sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		# load a manager to deal with test data

		cpkt = restore_session(self._sess,saver,self._config.models_path)




	def start(self,host,port,config_path=None,resolution=None):

		self.socket_control = connect(host,port+1)
		print "CONNECTED TO CONTROL HOST"
		self.gta = StreamClient(host,port)

		self.gta.run()
		print "Start The streaming"
		self._resolution = resolution
		self._start_time = time.time()
		self._accumulated_distance = 0.0


	def _get_direction_buttons(self):
		#with suppress_stdout():if keys[K_LEFT]:
		keys=pygame.key.get_pressed()

		if( keys[K_s]):

			self._left_button = False   
			self._right_button = False
			self._straight_button = False

		if( keys[K_a]):

			self._left_button = True    
			self._right_button = False
			self._straight_button = False


		if( keys[K_d]):
			self._right_button = True
			self._left_button = False
			self._straight_button = False

		if( keys[K_w]):

			self._straight_button = True
			self._left_button = False
			self._right_button = False

		return [self._left_button,self._right_button,self._straight_button]

	def compute_direction(self,pos,ori,goal_pos,goal_ori):  # This should have maybe some global position... GPS stuff
		if self._use_planner:

			return self.planner.get_next_command(pos,ori,goal_pos,goal_ori)

		else:
			# BUtton 3 has priority
			if 'Control' not in set(self._config.inputs_names):
				return None

			button_vec = self._get_direction_buttons()
			if sum(button_vec) == 0: # Nothing
				return 2
			elif button_vec[0] == True: # Left
				return 3
			elif button_vec[1] == True: # RIght
				return 4
			else:
				return 5


	def get_recording(self):

		return False




	def compute_action(self,sensor,speed,direction = 2.0):


		capture_time = time.time()

		sensor = sensor[130:530,:,:]

		sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

		image_input = sensor.astype(np.float32)

		#print future_image

		image_input = image_input - self._mean_image
		#print "2"
		image_input = np.multiply(image_input, 1.0 / 127.0)


		steer,acc,brake = machine_output_functions.branched_speed_4cmd(image_input,speed,direction,self._config,self._sess,self._train_manager)



		control = Control()
		control.steer = steer
		control.gas =acc
		control.brake =brake

		control.hand_brake = 0
		control.reverse = 0

		return control


	def _filter_direction(self,direction_dist):

		other_dist = 0.0



		if direction_dist[0] not in set([1.0,2.0,3.0,4.0,5.,6.,7.,8.,9.,10.0,11.0,0.0]):
			other_dist = direction_dist[0]
			direction_dist[0] = 2.0


		# direction[0] = 2.0

		#dist_to_goal = math.sqrt(( goal[0]- position[0]) *(goal[0] - position[0]) + (goal[1] - position[1]) *(goal[1] - position[1])) 
		#i#f dist_to_goal < 20.0:
		#	other_dist = 0.0


	
		return direction_dist[0],other_dist
  
  

	def get_number_completions(self):
		return self._number_completions

	def get_sensor_data(self):


		message = self.gta.get_message()


		image = frame2numpy(message['frame'],self._resolution)
		reward = Reward()
		reward.position = message['location']
		reward.speed =  message['speed']
		reward.yaw_rate = message['yawRate']
		reward.yaw = message['yaw']
		reward.collided = message['collided']
		reward.lane = message['lane']
		reward.reseted = message['reset']
		reward.goal = message['goal']

		reward.time_stamp = time.time() - self._start_time


		if reward.reseted > self._current_epi: # reseted


			goal = self._current_goal
			position = self._current_pos
			dist_to_goal = math.sqrt(( goal[0]- position[0]) *(goal[0] - position[0]) + (goal[1] - position[1]) *(goal[1] - position[1])) 
			if dist_to_goal < 25.0:
				self._number_completions +=1

		else:  # we can compute how much the car move because it has not reseted
			goal = reward.position
			position = self._current_pos
			self._accumulated_distance += math.sqrt(( goal[0]- position[0]) *(goal[0] - position[0]) + (goal[1] - position[1]) *(goal[1] - position[1])) 	


		self._current_goal = reward.goal
		self._current_pos = reward.position
		self._current_epi = reward.reseted
		self._current_stamp = reward.time_stamp

		
		direction_dist = message['direction']



		end_message = message['end']
		# End the system right here
		if end_message:
			self.stream_client._exit=True
			time.sleep(0.1)
			exit()

		direction,other_dist = self._filter_direction(direction_dist)

		reward.direction = direction
		reward.dist_to_goal = direction_dist[1]
		 	
		return reward,image


	def compute_perception_activations(self,sensor,speed):




		sensor = sensor[130:530,:,:]

		sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

		image_input = sensor.astype(np.float32)

		#print future_image

		image_input = image_input - self._mean_image
		#print "2"
		image_input = np.multiply(image_input, 1.0 / 127.0)


		vbp_image =  machine_output_functions.vbp(image_input,speed,self._config,self._sess,self._train_manager)

		#min_max_scaler = preprocessing.MinMaxScaler()
		#vbp_image = min_max_scaler.fit_transform(np.squeeze(vbp_image))

		print vbp_image.shape
		return 0.5*grayscale_colormap(np.squeeze(vbp_image),'jet') + 0.5*image_input

		

	def act(self,action):


		if action.brake > 0.01:
			gas_brake =  -0.2400-(action.brake*(1-0.2400))
		else:
			gas_brake = 0.2400+(action.gas*(1-0.2400))


		send_message_str(self.socket_control,str(action.steer)+';'+str(gas_brake))


