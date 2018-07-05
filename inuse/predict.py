

import traceback

import sys


sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla_interface')
sys.path.append('drive_interfaces/gta_interface')
sys.path.append('drive_interfaces/deeprc_interface')
sys.path.append('drive_interfaces/carla_interface/carla_client')

sys.path.append('drive_interfaces/carla_interface/carla_client/protoc')
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('structures')
sys.path.append('evaluation')

import h5py

import math
import argparse
from noiser import Noiser

import datetime

from screen_manager import ScreenManager

import numpy as np
import os
import time

class Control:
    steer = 0
    gas = 0
    brake =0
    hand_brake = 0
    reverse = 0


#from config import *
#from eConfig import *
from drawing_tools import *
from extra import *

clock = pygame.time.Clock()
def frame2numpy(frame, frameSize):
	return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))




def predict(experiment_name,drive_config,name = None,memory_use=1.0):
	#host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name

	

	from driver_machine import DriverMachine
	driver = DriverMachine("0",experiment_name,drive_config,memory_use)

	recorder = None

	noiser = Noiser(drive_config.noise)

	print 'before starting'
	driver.start()
	first_time = True
	if drive_config.show_screen:
		screen_manager = ScreenManager()
		screen_manager.start_screen(drive_config.resolution,drive_config.number_screens,drive_config.scale_factor)

	driver.use_planner =False



	direction = 2

	positions_to_test =  range(0,300)

	images= [np.array([200,88,3]),np.array([200,88,3]),np.array([200,88,3])]
	actions = [0,0,0]

	path = '/home/fcodevil/Datasets/DRC/DRC_O3/SeqTrain/'

	iteration = 0
	try:
		for  h_num in positions_to_test:
			capture_time  = time.time()
			data = h5py.File(path+'data_'+ str(h_num).zfill(5) +'.h5', "r+")

			for i in range(0,198,3):
				img_1 = np.array(data['images_center'][i]).astype(np.uint8)


				img_2 = np.array(data['images_center'][i+1]).astype(np.uint8)


				img_3 = np.array(data['images_center'][i+2]).astype(np.uint8)

				print int(data['targets'][i][49])




				images[int(data['targets'][i][49])] =  img_1
				images[int(data['targets'][i+1][49])] =  img_2
				images[int(data['targets'][i+2][49])] =  img_3

				action_1 = Control()
				action_1.steer = data['targets'][i][0]
				action_1.gas =data['targets'][i][1]
				action_2 = Control()
				action_2.steer = data['targets'][i+1][0]
				action_2.gas =data['targets'][i+1][1]
				action_3 = Control()
				action_3.steer = data['targets'][i+2][0]
				action_3.gas =data['targets'][i+2][1]

				actions[int(data['targets'][i][49])] =action_1
				actions[int(data['targets'][i+1][49])] =action_2
				actions[int(data['targets'][i+2][49])] =action_3


				
				# Compute now the direction
				if drive_config.show_screen:
					for event in pygame.event.get(): # User did something
						if event.type == pygame.QUIT: # If user clicked close
							done=True # Flag that we are done so we exit this loop





				actions_pred = driver.compute_action(images[drive_config.middle_camera],data['targets'][i][51],data['targets'][i+2][8])

				



				if drive_config.show_screen:
					if drive_config.interface == "Carla":
						for j in range(drive_config.number_screens):
							screen_manager.plot_driving_interface( capture_time,np.copy(images[j]),\
								actions[j],action_noisy,rewards.direction,recording and (drifting_time == 0.0 or  will_drift),\
								drifting_time,will_drift,rewards.speed,0,0,i) #

					elif drive_config.interface == "GTA":

						dist_to_goal = math.sqrt(( rewards.goal[0]- rewards.position[0]) *(rewards.goal[0] - rewards.position[0]) + (rewards.goal[1] - rewards.position[1]) *(rewards.goal[1] - rewards.position[1]))
						
						image = image[:, :, ::-1]
						screen_manager.plot_driving_interface( capture_time,np.copy(images),	action,action_noisy,\
							rewards.direction,recording and (drifting_time == 0.0 or  will_drift),drifting_time,will_drift\
							,rewards.speed,0,0,None,rewards.reseted,driver.get_number_completions(),dist_to_goal,0) #

					elif drive_config.interface == "DeepRC":
						for key,value in drive_config.cameras_to_plot.iteritems():
							screen_manager.plot_driving_interface( capture_time,np.copy(images[key]),\
								actions[key],actions_pred[key],data['targets'][i+2][8],0,\
								0,0,data['targets'][i][51],0,0,value) #
					else:
						print "Not supported interface"
						pass

				
				if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:

					image_vbp =driver.compute_perception_activations(images[drive_config.middle_camera],data['targets'][i][51])

					screen_manager.plot_image(image_vbp,1)

				time.sleep(0.1)
				 	
				iteration +=1
	except:
		traceback.print_exc()

	finally:

		#driver.write_performance_file(path,folder_name,iteration)
		pygame.quit()


