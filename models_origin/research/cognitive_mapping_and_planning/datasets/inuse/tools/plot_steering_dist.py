#!/usr/bin/env python
import sys

sys.path.append('utils')
sys.path.append('configuration')


import argparse
import numpy as np
import h5py
import pygame
#import readchar
#import json
#from keras.models import

from PIL import Image
import matplotlib.pyplot as plt
import pygame
import math
from drawing_tools import *
import time
import scipy
import os
import scipy
from collections import deque
from skimage.transform import resize
import seaborn as sns
sns.set(color_codes=True)

sys.path.append('drive_interfaces')

from screen_manager import ScreenManager

class Control:
    steer = 0
    throttle =0
    brake =0
    hand_brake = 0
    reverse = 0



# Configurations for this script

path = '/media/matthias/7E0CF8640CF818BB/Github/ModularEnd2End/Desktop/20171025_WeatherN1_1_clean/'

sensors = {'RGB':3,'labels':3,'depth':3}
resolution = [200,88]
camera_id_position = 25
direction_position = 24
speed_position = 10
number_of_seg_classes = 5
classes_join = {0:2,1:2,2:2,3:2,5:2,12:2,9:2,11:2,4:0,10:1,8:3,6:3,7:4}

def join_classes(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in join_dic.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value


  return compressed_labels_image

def join_classes_for(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  print compressed_labels_image.shape
  for i in range(labels_image.shape[0]):
    for j in range(labels_image.shape[1]):
      compressed_labels_image[i,j,0] = join_dic[labels_image[i,j,0]]



  return compressed_labels_image



# ***** main loop *****
if __name__ == "__main__":

  first_time = True
  count =0
  steering_pred =[]
  steering_gt =[]
  index_vec =[]
  #initial_positions =[20,25,48,68,79,105,108,120,130]
  #positions_to_test = []
  #for i in initial_positions:
  #  positions_to_test += range(i-1,i+2)
  positions_to_test = range(0,len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))


  screen = ScreenManager()

  image_queue = deque()

  actions_queue = deque()


  # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
  # But maybe a more arbitrary configuration may be useful

  screen.start_screen([resolution[0],resolution[1]],[sensors['RGB'],2],2)
  ts =[]
  images= [np.zeros([resolution[1],resolution[0],3])]* sensors['RGB']
  labels= [np.zeros([resolution[1],resolution[0],1])]* sensors['labels']
  depths= [np.zeros([resolution[1],resolution[0],3])]* sensors['depth']
  actions = [Control()] * sensors['RGB'] 
  actions_noise = [Control()]  * sensors['RGB']
  
  
  for h_num in positions_to_test:

    print " SEQUENCE NUMBER ",h_num
    try:
      data = h5py.File(path+'data_'+ str(h_num).zfill(5) +'.h5', "r")
    except Exception as e:
      print e
      continue
      pass

    for i in range(0,200,1):


      speed = data['targets'][i][speed_position]


      steer = data['targets'][i][0]
      steering_pred.append(steer)
      index_vec.append(h_num)


      #actions[0].steer +=min(4*(math.atan((0.26*car_lenght)/(time_use*speed+0.05)))/3.1415,0.2)
      #actions[2].steer -=min(4*(math.atan((0.26*car_lenght)/(time_use*speed+0.05)))/3.1415,0.2)
      #print 'Steer ',actions[0].steer
      #print 'Throttle ',actions[0].throttle
      #print 'Brake ',actions[0].brake

     
      #for j in range(sensors['depth']):
      #  #print j
        
      #  screen.plot_camera(depths[j] ,[j,2])



      #pygame.display.flip()
      #time.sleep(0.05)



  plt.plot(index_vec,steering_pred)
  plt.show()
  #save_gta_surface(gta_surface)




