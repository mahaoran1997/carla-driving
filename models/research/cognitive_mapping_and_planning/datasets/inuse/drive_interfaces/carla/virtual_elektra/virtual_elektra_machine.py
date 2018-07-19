
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
from ConfigParser import ConfigParser



import pygame

sys.path.append('../train')
from pygame.locals import *

from carla import CARLA
from carla import Measurements
from carla import Control
from carla import Planner

from sklearn import preprocessing

from codification import *

from training_manager import TrainManager
import machine_output_functions
#from Runnable import *
from driver import *
from drawing_tools import *
import copy
import random

Measurements.noise = property(lambda self: 0)


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



def convert_to_car_coord(goal_x,goal_y,pos_x,pos_y,car_heading_x,car_heading_y):

  start_to_goal = (goal_x- pos_x,goal_y - pos_y)


  car_goal_x = -(-start_to_goal[0]*car_heading_y + start_to_goal[1]*car_heading_x)
  car_goal_y = start_to_goal[0]*car_heading_x + start_to_goal[1]*car_heading_y
 
  return [car_goal_x,car_goal_y]
      



class VirtualElektraMachine(Driver):


  def __init__(self,gpu_number="0",experiment_name ='None',driver_conf=None,memory_fraction=0.9,\
    trained_manager =None,session =None,config_input=None):



    Driver.__init__(self)

    if  trained_manager ==None:
      
      conf_module  = __import__(experiment_name)
      self._config = conf_module.configInput()
      
      config_gpu = tf.ConfigProto()
      config_gpu.gpu_options.visible_device_list=gpu_number

      config_gpu.gpu_options.per_process_gpu_memory_fraction=memory_fraction
      self._sess = tf.Session(config=config_gpu)
        
      self._train_manager =  load_system(conf_module.configTrain())


      self._sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(tf.global_variables())
      cpkt = restore_session(self._sess,saver,self._config.models_path)

    else:
      self._train_manager = trained_manager
      self._sess = session
      self._config =config_input




    if self._train_manager._config.control_mode == 'goal':
      self._select_goal = conf_module.configTrain().select_goal


    self._control_function =getattr(machine_output_functions, self._train_manager._config.control_mode )


    self._image_cut = driver_conf.image_cut




    # load a manager to deal with test data
    self.use_planner = driver_conf.use_planner
    if driver_conf.use_planner:
      self.planner = Planner('drive_interfaces/carla_interface/' + driver_conf.city_name  + '.txt',\
        'drive_interfaces/carla_interface/' + driver_conf.city_name + '.png')

    self._host = driver_conf.host
    self._port = driver_conf.port
    self._config_path = driver_conf.carla_config
    self._resolution = driver_conf.resolution


    self._straight_button = False
    self._left_button = False
    self._right_button = False
    self._recording= False




  def start(self):

    # You start with some configurationpath

    self.carla =CARLA(self._host,self._port)
    self.positions= self.carla.loadConfigurationFile(self._config_path)

    self._current_goal = random.randint(0,len(self.positions)-1)
    position_reset =random.randint(0,len(self.positions)-1)

    self.carla.newEpisode(position_reset)

    self._target = random.randint(0,len(self.positions))
    self._start_time = time.time()


    

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


  def compute_goal(self,pos,ori): #Return the goal selected
    pos,point = self.planner.get_defined_point(pos,ori,(self.positions[self._target][0],self.positions[self._target][1],22),(1.0,0.02,-0.001),1+self._select_goal)
    return convert_to_car_coord(point[0],point[1],pos[0],pos[1],ori[0],ori[1])
    

  def compute_direction(self,pos,ori):  # This should have maybe some global position... GPS stuff
    
    if self._train_manager._config.control_mode == 'goal':
      return self.compute_goal(pos,ori)

    elif self.use_planner:

      command,made_turn,completed = self.planner.get_next_command(pos,ori,(self.positions[self._target].location.x,self.positions[self._target].location.y,22),(1.0,0.02,-0.001))
      return command

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

  def get_reset(self):
    return False




  def new_episode(self,initial_pos,target,cars,pedestrians,weather):


    config = ConfigParser()

    config.read(self._config_path)
    config.set('CARLA/LevelSettings','NumberOfVehicles',cars)

    config.set('CARLA/LevelSettings','NumberOfPedestrians',pedestrians)

    config.set('CARLA/LevelSettings','WeatherId',weather)

    # Write down a temporary init_file to be used on the experiments
    temp_f_name = 's' +str(initial_pos)+'_e'+ str(target) + "_p" +\
     str(pedestrians)+'_c' + str(cars)+"_w" + str(weather) +\
     '.ini'

    with open(temp_f_name, 'w') as configfile:
      config.write(configfile)


    positions = self.carla.requestNewEpisode(temp_f_name)
      
    self.carla.newEpisode(initial_pos)
    self._target = target


  def get_all_turns(self,data,target):
    rewards = data[0]
    sensor = data[2][0]
    speed = rewards.speed
    return self.planner.get_all_commands((rewards.player_x,rewards.player_y,22),(rewards.ori_x,rewards.ori_y,rewards.ori_z),\
      (target[0],target[1],22),(1.0,0.02,-0.001))

  #### TO BE DEPRECATED , WE DONT NEED TWO FUNCTIONS ....
  def run_step(self,data,target):

    rewards = data[0]
    sensor = data[2][0]
    speed = rewards.speed
    direction,made_turn,completed = self.planner.get_next_command((rewards.player_x,rewards.player_y,22),(rewards.ori_x,rewards.ori_y,rewards.ori_z),\
      (target[0],target[1],22),(1.0,0.02,-0.001))
    #pos = (rewards.player_x,rewards.player_y,22)
    #ori =(rewards.ori_x,rewards.ori_y,rewards.ori_z)
    #pos,point = self.planner.get_defined_point(pos,ori,(target[0],target[1],22),(1.0,0.02,-0.001),self._select_goal)
    #direction = convert_to_car_coord(point[0],point[1],pos[0],pos[1],ori[0],ori[1])

    capture_time = time.time()

    sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:]

    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    #print future_image
    #print direction
    #print "2"
    image_input = np.multiply(image_input, 1.0 / 255.0)


    steer,acc,brake = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)


    if brake < 0.1:
      brake =0.0
    if acc> 2*brake:
      brake =0.0
      
    control = Control()
    control.steer = steer
    control.throttle =acc
    control.brake =brake

    control.hand_brake = 0
    control.reverse = 0


    #### DELETE THIS VERSION NOT COMMITABLE
    made_turn = 0
    completed = 0

    return control,made_turn,completed

  def compute_action(self,sensor,speed,direction=None):
    
    capture_time = time.time()

    if capture_time - self._start_time >400:

      self._target = random.randint(0,len(self.positions))
      self._start_time = time.time()  
      

    if direction == None:
      direction = self.compute_direction((0,0,0),(0,0,0))

    sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:3]
    sensor = sensor[:, :, ::-1]

    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    #print future_image

    #print "2"

   
    image_input = np.multiply(image_input, 1.0 / 255.0)


    steer,acc,brake = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)




    control = Control()
    control.steer = steer



    if control.steer > 0.3:
      control.steer =1.0
    elif control.steer < -0.3:
      control.steer = -1.0
    else:
      control.steer = 0.0

    control.throttle =1.0
    control.brake =0.0
    # print brake

    control.hand_brake = 0
    control.reverse = 0



    return control

  
  # The augmentation should be dependent on speed



  def get_sensor_data(self):
    measurements= self.carla.getMeasurements()
    self._latest_measurements = measurements
    player_data =measurements['PlayerMeasurements']
    pos = [player_data.transform.location.x,player_data.transform.location.y,22]
    ori = [player_data.transform.orientation.x,player_data.transform.orientation.y,player_data.transform.orientation.z]
    

    direction = 2.0


    return measurements,direction



  def compute_perception_activations(self,sensor,speed):




    sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:3]
    sensor = sensor[:, :, ::-1]
    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    #print future_image

    #print "2"
    image_input = np.multiply(image_input, 1.0 / 255.0)


    vbp_image =  machine_output_functions.vbp_nospeed(image_input,self._config,self._sess,self._train_manager)

    min_max_scaler = preprocessing.MinMaxScaler()
    vbp_image = min_max_scaler.fit_transform(np.squeeze(vbp_image))

    # print vbp_image
    #print vbp_image
    #print grayscale_colormap(np.squeeze(vbp_image),'jet')

    vbp_image_3 = np.copy(image_input)
    vbp_image_3[:,:,0] = vbp_image
    vbp_image_3[:,:,1] = vbp_image
    vbp_image_3[:,:,2] = vbp_image
    #print vbp_image

    return  0.4*grayscale_colormap(np.squeeze(vbp_image),'inferno') + 0.6*image_input
  
  def act(self,action):


    self.carla.sendCommand(action)


  def stop(self):

    self.carla.stop()

