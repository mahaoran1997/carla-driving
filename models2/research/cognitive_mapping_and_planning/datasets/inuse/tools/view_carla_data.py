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
    gas =0
    brake =0
    hand_brake = 0
    reverse = 0

#gta_surface = get_gta_map_surface()


# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Path viewer')
  #parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
  parser.add_argument('--dataset', type=str, default="2_SegData_2", help='Dataset/video clip name')
  args = parser.parse_args()

  dataset = args.dataset
  first_time = True
  count =0
  steering_pred =[]
  steering_gt =[]

  positions_to_test =  range(0,28)
  #positions_to_test = [93,104,170,173,229,245,283,397,413,425,565,581,591]
  #positions_to_test = range(0,660)
  #positions_to_test = [617,618,619,620,622,623,624,636,637,638,639]
  #positions_to_test =  [637,638]
  #positions_to_test = [55,108,109,353,410,411,426,441,442]
  #positions_to_test = [656,657,675,676,854,855,859,860,861,902]
  path = '/media/matthias/7E0CF8640CF818BB/Github/Desktop/2_SegData_1/'


  screen = ScreenManager()

  image_queue = deque()

  speed_list = []
  actions_queue = deque()
  speed_list_noise = []
  just_noise = []
  screen.start_screen([800,600],1,1)
  ts =[]
  images= [np.array([800,600,3]),np.array([800,600,3]),np.array([800,600,3])] #200x88
  actions = [Control(),Control(),Control()]
  actions_noise = [Control(),Control(),Control()]

  

  for h_num in positions_to_test:

    print " SEQUENCE NUMBER ",h_num
    data = h5py.File(path+'data_'+ str(h_num).zfill(5) +'.h5', "r")

    #redata = h5py.File('/media/adas/012B4138528FF294/NewGTA/redata_'+ str(h_num).zfill(5) +'.h5', "r")
    #print log.keys()


    
    #save_data_stats = '../../../Data/Udacity/'

    



    # skip to highway
    for i in range(0,198,3):




      #img = cam['X'][log['cam1_ptr'][i]].swapaxes(0,2).swapaxes(0,1)

      img_1 = np.array(data['images_center'][i]).astype(np.uint8)


      img_2 = np.array(data['images_center'][i+1]).astype(np.uint8)


      img_3 = np.array(data['images_center'][i+2]).astype(np.uint8)


      images[int(data['targets'][i][26])] =  img_1
      images[int(data['targets'][i+1][26])] =  img_2
      images[int(data['targets'][i+2][26])] =  img_3

      action_1 = Control()
      action_1.steer = data['targets'][i][0]
      action_1.gas =data['targets'][i][1]
      action_2 = Control()
      action_2.steer = data['targets'][i+1][0]
      action_2.gas =data['targets'][i+1][1]
      action_3 = Control()
      action_3.steer = data['targets'][i+2][0]
      action_3.gas =data['targets'][i+2][1]
      #print  data['targets'][i][20]
      actions[int(data['targets'][i][26])] =action_1
      actions[int(data['targets'][i+1][26])] =action_2
      actions[int(data['targets'][i+2][26])] =action_3


      action_1 = Control()
      action_1.steer = data['targets'][i][5]
      action_1.gas =data['targets'][i][6]
      action_2 = Control()
      action_2.steer = data['targets'][i+1][5]
      action_2.gas =data['targets'][i+1][6]
      action_3 = Control()
      action_3.steer = data['targets'][i+2][5]
      action_3.gas =data['targets'][i+2][6]
      #print  data['targets'][i][20]
      actions_noise[int(data['targets'][i][26])] =action_1
      actions_noise[int(data['targets'][i+1][26])] =action_2
      actions_noise[int(data['targets'][i+2][26])] =action_3
      direction = data['targets'][i+2][22]

      speed = data['targets'][i+2][10]
      time_use =  1.0
      car_lenght = 6
      actions[0].steer +=min(4*(math.atan((0.26*car_lenght)/(time_use*speed+0.05)))/3.1415,0.2)
      actions[2].steer -=min(4*(math.atan((0.26*car_lenght)/(time_use*speed+0.05)))/3.1415,0.2)
      print " Steer Left MIDDLE Right "
      print actions[0].steer
      print actions[1].steer
      print actions[2].steer

      for j in range(1):
        screen.plot3camrcnoise( images[j],\
              actions[j].steer,-actions[j].steer + actions_noise[j].steer,actions_noise[j].steer,j) #

      time.sleep(0.0)

      image_result = Image.fromarray(images[0])
      image_result.save('temp/image' + str(i) + '.png')
      speed_list.append((actions[0].steer))

      speed_list_noise.append((actions_noise[0].steer))
      just_noise.append((-actions[0].steer + actions_noise[0].steer))
      ts.append((data['targets'][i][20] - 93532.0)/1000.0)
      #reimg = np.array(redata['images_center'][i])
      #recontrol_input = np.array(redata['control'][i][1])
      #print img
      #img = img*255
      #print img

  print speed_list
  print speed_list_noise
  print just_noise
  print ts
  plt.plot(range(0,len(speed_list)),speed_list,'g',range(0,len(speed_list)),speed_list_noise,'b',range(0,len(speed_list)),just_noise,'r')
  
  plt.show()
  #save_gta_surface(gta_surface)




