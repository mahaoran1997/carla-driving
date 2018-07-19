
import sys
import pygame

sys.path.append('../')
import socket
from client import *

from socket_util import *

from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from contextlib import contextmanager
import sys, os
import math
import scipy

# TODO: Make a joystick interface, so we can easily change the joystick



class GTAHuman(object):


	def __init__(self):

		self._straight_button = False
		self._left_button = False
		self._right_button = False
		self._recording= False
		self._rear = False
		self._current_epi = 0
		self._current_pos =[0,0,0]
		self._current_goal=[100,100]
		self._number_completions =0



	def start(self,host,port,config_path=None,resolution=None):


		#try:
		self.socket_control = connect(host,port+1)
		print "CONNECTED TO CONTROL HOST"
		self.gta = StreamClient(host,port)

		self.gta.run()
		print "Start The streaming"
		self._resolution = resolution
		self._start_time = time.time()
		self._accumulated_distance = 0.0


		#except:
		#	print 'ERROR: Failed to connect to DrivingServer'
		#else:
		#	print 'Successfully connected to DrivingServer'

		# Now start the joystick, the actual controller 

		
		joystick_count = pygame.joystick.get_count()
		if joystick_count >1:
			print "Please Connect Just One Joystick"
			raise 



		self.joystick = pygame.joystick.Joystick(0)
		self.joystick.init()

	def _get_direction_buttons(self):
		#with suppress_stdout():
		if( self.joystick.get_button( 6)):
			
			self._left_button = False		
			self._right_button = False
			self._straight_button = False

		if( self.joystick.get_button( 5 )):
			
			self._left_button = True		
			self._right_button = False
			self._straight_button = False


		if( self.joystick.get_button( 4 )):
			self._right_button = True
			self._left_button = False
			self._straight_button = False

		if( self.joystick.get_button( 7 )):

			self._straight_button = True
			self._left_button = False
			self._right_button = False

			 	
		return [self._left_button,self._right_button,self._straight_button]


	def compute_direction(self):  # This should have maybe some global position... GPS stuff
		
		# BUtton 3 has priority

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


		if( self.joystick.get_button( 9 )):

			self._recording =True
		if( self.joystick.get_button( 8 )):
			self._recording=False

		return self._recording

	def get_reset(self):

		return False


	def get_noise(self,noise):
		if( self.joystick.get_button( 5 )):
			if noise ==True:
				return False



			return noise 

	def compute_action(self,sensor,direction,speed):


		""" Get Steering """

		steering_axis = self.joystick.get_axis(0)

		acc_axis = self.joystick.get_axis(2)

		brake_axis = self.joystick.get_axis(3)

		if( self.joystick.get_button( 3 )):

			self._rear =True
		if( self.joystick.get_button( 2 )):
			self._rear=False


		control = Control()
		control.steer = steering_axis
		control.gas = -(acc_axis -1)/2.0
		control.brake = -(brake_axis -1)/2.0
		if control.brake < 0.01:
			control.brake = 0.0
		
			
		control.hand_brake = 0
		control.reverse = self._rear




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

	
	def act(self,action):




		if action.brake > 0.01:
			gas_brake =  -0.2400-(action.brake*(1-0.2400))
		else:
			gas_brake = 0.2400+(action.gas*(1-0.2400))

		if self._rear:
			gas_brake = -gas_brake

		send_message_str(self.socket_control,str(action.steer)+';'+str(gas_brake))


	def write_performance_file(self,path,folder_name,iterations):

		outfile =open(path + folder_name + '/' + folder_name + '.txt' ,'a+')
		outfile.write("%f,%d,%f,%f\n" % (self._current_stamp ,iterations,self._accumulated_distance,float(self._number_completions)/float(self._current_epi)))			
		outfile.close()
