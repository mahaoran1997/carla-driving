
import sys
import os

import socket
import scipy
import re
from PIL import Image
import math

from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
import tensorflow as tf
import time
from ConfigParser import ConfigParser

import math
import pygame
from pygame.locals import *
sys.path.append('../train')

from carla import CARLA
from carla import Measurements
from carla import Control
from carla.agent import *
from carla import Planner

from sklearn import preprocessing

from codification import *

from training_manager import TrainManager
import machine_output_functions
from Runnable import *
from driver import *
from drawing_tools import *
import copy
import random

slim = tf.contrib.slim

Measurements.noise = property(lambda self: 0)



number_of_seg_classes = 5
classes_join = {0:2,1:2,2:2,3:2,5:2,12:2,9:2,11:2,4:0,10:1,8:3,6:3,7:4}


def join_classes(labels_image,labels_mapping):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in labels_mapping.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value
  return compressed_labels_image



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

  training_manager= TrainManager(config,None)

  if hasattr(config, 'rgb_seg_network_one_hot'):
    training_manager.build_rgb_seg_network_one_hot()
    print("Bulding: rgb_seg_network_one_hot")

  else:
    if hasattr(config, 'seg_network_gt_one_hot'):
      training_manager.build_seg_network_gt_one_hot()
      print("Bulding: seg_network_gt_one_hot")

    else:
      if hasattr(config, 'seg_network_gt_one_hot_join'):
        training_manager.build_seg_network_gt_one_hot_join()
        print("Bulding: seg_network_gt_one_hot_join")

      else:
        if hasattr(config, 'rgb_seg_network_enet'):
          training_manager.build_rgb_seg_network_enet()
          print("Bulding: rgb_seg_network_enet")

        else:
          if hasattr(config, 'rgb_seg_network_enet_one_hot'):
            training_manager.build_rgb_seg_network_enet_one_hot()
            print("Bulding: rgb_seg_network_enet_one_hot")

          else:
            if hasattr(config, 'seg_network_enet_one_hot'):
              training_manager.build_seg_network_enet_one_hot()
              print("Bulding: seg_network_enet_one_hot")

            else:
              if hasattr(config, 'seg_network_erfnet_one_hot'):
                training_manager.build_seg_network_erfnet_one_hot() 
                print("Bulding: seg_network_erfnet_one_hot")

              else:
                training_manager.build_network()
                print("Bulding: standard_network")



  """ Initializing Session as variables that control the session """
 
  return training_manager



def convert_to_car_coord(goal_x,goal_y,pos_x,pos_y,car_heading_x,car_heading_y):

  start_to_goal = (goal_x- pos_x,goal_y - pos_y)

  car_goal_x = -(-start_to_goal[0]*car_heading_y + start_to_goal[1]*car_heading_x)
  car_goal_y = start_to_goal[0]*car_heading_x + start_to_goal[1]*car_heading_y
 
  return [car_goal_x,car_goal_y]
      



class CarlaMachine(Runnable,Driver):


  def __init__(self,gpu_number="0",experiment_name ='None',driver_conf=None,memory_fraction=0.9,\
    trained_manager =None,session =None,config_input=None):


    #use_planner=False,graph_file=None,map_file=None,augment_left_right=False,image_cut = [170,518]):

    Driver.__init__(self)

    if  trained_manager ==None:
      
      conf_module  = __import__(experiment_name)
      self._config = conf_module.configInput()
      
      config_gpu = tf.ConfigProto()
      config_gpu.gpu_options.visible_device_list=gpu_number

      config_gpu.gpu_options.per_process_gpu_memory_fraction=memory_fraction
      self._sess = tf.Session(config=config_gpu)
        
      self._train_manager =  load_system(conf_module.configTrain())
      self._config.train_segmentation = False



      self._sess.run(tf.global_variables_initializer())

      if conf_module.configTrain().restore_seg_test:
        if  self._config.segmentation_model != None:
          exclude = ['global_step']

          variables_to_restore = slim.get_variables(scope=str(self._config.segmentation_model_name))

          saver = tf.train.Saver(variables_to_restore,max_to_keep=0)

          seg_ckpt = restore_session(self._sess,saver,self._config.segmentation_model)

        variables_to_restore = list(set(tf.global_variables()) - set(slim.get_variables(scope=str(self._config.segmentation_model_name))))

      else:
        variables_to_restore = tf.global_variables()

      saver = tf.train.Saver(variables_to_restore)
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
    import os
    dir_path = os.path.dirname(__file__)
    if driver_conf.use_planner:
      self.planner = Planner(dir_path +'/../carla_client/carla/planner/'+ driver_conf.city_name  + '.txt',\
        dir_path +'/../carla_client/carla/planner/' + driver_conf.city_name + '.png')

    self._host = driver_conf.host
    self._port = driver_conf.port
    self._config_path = driver_conf.carla_config
    self._resolution = driver_conf.resolution


    self._straight_button = False
    self._left_button = False
    self._right_button = False
    self._recording= False
    self._start_time =0



  def start(self):

    # You start with some configurationpath

    self.carla =CARLA(self._host,self._port)
      
    self.positions =  self.carla.loadConfigurationFile(self._config_path)


    self.carla.newEpisode(random.randint(0,len(self.positions)))

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


  def get_vec_dist (self, x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst,y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0]**2 + vec[1]**2)
    return vec/dist, dist

  def get_angle (self, vec_dst, vec_src):
    angle = math.atan2(vec_dst[1]-vec_src[1],vec_dst[0]-vec_src[0])
    if angle > math.pi:
      angle -= 2*math.pi
    elif angle < -math.pi:
      angle += 2*math.pi
    return angle


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

  def run_step(self,measurements,target):



    direction,_ = self.planner.get_next_command((measurements['PlayerMeasurements'].transform.location.x,measurements['PlayerMeasurements'].transform.location.y,22),\
      (measurements['PlayerMeasurements'].transform.orientation.x,measurements['PlayerMeasurements'].transform.orientation.y,measurements['PlayerMeasurements'].transform.orientation.z),\
      (target.location.x,target.location.y,22),(1.0,0.02,-0.001))
    #pos = (rewards.player_x,rewards.player_y,22)
    #ori =(rewards.ori_x,rewards.ori_y,rewards.ori_z)
    #pos,point = self.planner.get_defined_point(pos,ori,(target[0],target[1],22),(1.0,0.02,-0.001),self._select_goal)
    #direction = convert_to_car_coord(point[0],point[1],pos[0],pos[1],ori[0],ori[1])
    print measurements['PlayerMeasurements'].forward_speed
    print direction
    sensors = []
    for name in self._config.sensor_names:
      if name =='rgb':
        sensors.append(measurements['BGRA'][0]) 
      elif name == 'labels':
        sensors.append(measurements['Labels'][0])
        
    control = self.compute_action(sensors,measurements['PlayerMeasurements'].forward_speed,direction)

    return control

  def compute_action(self,sensors,speed,direction=None):
    
    capture_time = time.time()


    if direction == None:
      direction = self.compute_direction((0,0,0),(0,0,0))

    sensor_pack =[]
    for i in range(len(self._config.sensor_names)): #sensors
      print ('Number of sensors: %s' %(len(self._config.sensor_names)))
      sensor = sensors[i]

      if self._config.sensor_names[i] =='rgb':

        sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:3]
        sensor = sensor[:, :, ::-1]
        sensor = scipy.misc.imresize(sensor,[self._config.sensors_size[i][0],self._config.sensors_size[i][1]])


      elif self._config.sensor_names[i] =='labels':

        sensor = sensor[self._image_cut[0]:self._image_cut[1],:,2] 

        sensor = scipy.misc.imresize(sensor,[self._config.sensors_size[i][0],self._config.sensors_size[i][1]],interp='nearest')

        if hasattr(self._config, 'labels_mapping'):
          print ("Labels with mapping")
          sensor = join_classes(sensor,self._config.labels_mapping) * int(255/(self._config.number_of_labels-1))
        else:
          sensor = join_classes(sensor,classes_join) * int(255/(number_of_seg_classes-1))



        #image_result = Image.fromarray(sensor)
        #image_result.save('image.png')
        sensor = sensor[:,:,np.newaxis]


      sensor_pack.append(sensor)

    
    if len(sensor_pack) > 1:

      print sensor_pack[0].shape
      print sensor_pack[1].shape
      image_input =  np.concatenate((sensor_pack[0],sensor_pack[1]),axis=2)

    else:
      image_input = sensor_pack[0]
      print sensor_pack[0].shape
    

    image_input = image_input.astype(np.float32)

    image_input = np.multiply(image_input, 1.0 / 255.0)


    #print "2"
    #print image_input.shape
    #print speed
    #print direction


    if (self._train_manager._config.control_mode == 'single_branch_wp'):

      steer,acc,brake,wp1angle,wp2angle = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)

      steer_pred = steer

      steer_gain = 0.8
      steer = steer_gain*wp1angle

      if steer > 0:

        #print ("SQRT")
        #steer = math.sqrt(steer)
        
        #print ("SQ")
        #steer = math.pow(steer,2)

        steer = min(steer,1)
      else:

        #print ("SQRT")
        #steer = - math.sqrt(-steer)

        #print ("SQ")
        #steer = - math.pow(-steer,2)

        steer = max(steer,-1)

      print('Predicted Steering: ',steer_pred, ' Waypoint Steering: ', steer)

    else:
      steer,acc,brake = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)



    if brake < 0.1:
      brake =0.0

    if acc > brake:
      brake =0.0
    if speed > 35.0 and brake == 0.0:
      acc=0.0
      
    control = Control()

    control.steer = steer

    control.throttle = acc 
    control.brake =brake
    # print brake

    control.hand_brake = 0
    control.reverse = 0



    return control#,machine_output_functions.get_intermediate_rep(image_input,speed,self._config,self._sess,self._train_manager)

  
  # The augmentation should be dependent on speed



  def get_sensor_data(self):
    measurements= self.carla.getMeasurements()
    self._latest_measurements = measurements
    player_data =measurements['PlayerMeasurements']
    pos = [player_data.transform.location.x,player_data.transform.location.y,22]
    ori = [player_data.transform.orientation.x,player_data.transform.orientation.y,player_data.transform.orientation.z]
    
    if self.use_planner:

      if sldist([player_data.transform.location.x,player_data.transform.location.y],[self.positions[self.episode_config[1]].location.x,self.positions[self.episode_config[1]].location.y]) < self._dist_to_activate:

        self._reset()

      print 'Selected Position ',self.episode_config[1],'from len ', len(self.positions)
      direction,_ = self.planner.get_next_command(pos,ori,[self.positions[self.episode_config[1]].location.x,self.positions[self.episode_config[1]].location.y,22],(1,0,0))
      print direction 
    else:
      direction = 2.0


    return measurements,direction


  def compute_perception_activations(self,sensor,speed):

    #sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:]

    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    image_input = np.multiply(image_input, 1.0 / 255.0)

    #vbp_image =  machine_output_functions.vbp(image_input,speed,self._config,self._sess,self._train_manager)
    vbp_image =  machine_output_functions.seg_viz(image_input,speed,self._config,self._sess,self._train_manager)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #vbp_image = min_max_scaler.fit_transform(np.squeeze(vbp_image))

    #print vbp_image.shape
    return 0.4*grayscale_colormap(np.squeeze(vbp_image),'jet') + 0.6*image_input #inferno

  
  def act(self,action):


    self.carla.sendCommand(action)


  def stop(self):

    self.carla.stop()

