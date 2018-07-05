

import numpy as np
import cv2
import scipy


import copy
from driver import *
import logging
import tensorflow as tf
from training_manager import TrainManager
import machine_output_functions
import os
import time
import math
import random
class Control:
    steer = 0
    gas = 0
    brake =0
    hand_brake = 0
    reverse = 0



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



class DriverMachine(Driver):

  # Initializes all necessary shared data for this class
  def __init__(self,gpu_number,experiment_name,driver_conf,memory_fraction=0.95):
    
    Driver.__init__(self)


    self._augment_left_right = driver_conf.augment_left_right

    self._augmentation_camera_angles = driver_conf.camera_angle # The angle between the cameras used for augmentation and the central camera

    self.stream_id = 0 
    #Rate at which data stream from controller is received
    self.stream_rate = 10
    #Enable/Disable data stream
    self.stream_on = 1

    self.frame_counter = 0

    self._resolution = driver_conf.resolution
    self._image_cut = driver_conf.image_cut

    conf_module  = __import__(experiment_name)
    self._config = conf_module.configInput()

    config_gpu = tf.ConfigProto()

    config_gpu.gpu_options.per_process_gpu_memory_fraction=memory_fraction
    config_gpu.gpu_options.visible_device_list=gpu_number
    self._sess = tf.Session(config=config_gpu)

    #Input Vars
    # This are the channels from the rc controler that represent each of these variables


    self._mean_image = np.load('data_stats/'+ self._config.dataset_name + '_meanimage.npy')
    self._train_manager =  load_system(conf_module.configTrain())


    self._sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    self._control_function =getattr(machine_output_functions, self._train_manager._config.control_mode )


    cpkt = restore_session(self._sess,saver,self._config.models_path)

    #Output Vars


  # No Return
  def start(self):
    pass


  # @returns: a flag, if true, we should record the sensor data and actions.

  def get_recording(self):
    pass

  def compute_action(self,sensor,speed,direction):

    """ Get Steering """
    # receives from 1000 to 2000 
    # Just taking the center image to send to the network


    sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:]

    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    #print future_image

    image_input = image_input - self._mean_image
    #print "2"
    image_input = np.multiply(image_input, 1.0 / 127.0)


    steer,acc,brake = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)
    
    control = Control()
    control.steer = steer
    control.gas =acc
    control.brake =0

    control.hand_brake = 0
    control.reverse = 0

    if self._augment_left_right: # If augment data, we generate copies of steering for left and right
      control_left = copy.deepcopy(control)

      control_left.steer = self._adjust_steering(control_left.steer,self._augmentation_camera_angles,speed) # The angles are inverse.
      control_right = copy.deepcopy(control)

      control_right.steer = self._adjust_steering(control_right.steer,-self._augmentation_camera_angles,speed)

      return [control,control,control]

    else:
      return [control]

  # This function is were you should get the data and return it
  # @returns a vector [measurements,images] where:
  # -- measurements -> is a filled object of the class defined above in this file,  
  #     it should contain data from all collected sensors
  # -- images -> is the vector of collected images

  def get_sensor_data(self):

    pass

  # this is the function used to send the actions to the car
  # @param: an object of the control class.
  
  def act(self,control):
    pass

  def compute_perception_activations(self,sensor,speed):

    #sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:]

    #sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    image_input = np.multiply(image_input, 1.0 / 255.0)


    vbp_image =  machine_output_functions.vbp(image_input,speed,self._config,self._sess,self._train_manager)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #vbp_image = min_max_scaler.fit_transform(np.squeeze(vbp_image))

    #print vbp_image.shape
    return 0.5*grayscale_colormap(np.squeeze(vbp_image),'jet') + 0.5*image_input
