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
  parser.add_argument('--dataset', type=str, default="2016-06-08--11-46-01", help='Dataset/video clip name')
  args = parser.parse_args()





  dataset = args.dataset
  first_time = True
  count =0
  steering_pred =[]
  steering_gt =[]

  positions_to_test =  range(0,2060)
  #positions_to_test = [93,104,170,173,229,245,283,397,413,425,565,581,591]
  #positions_to_test = range(0,660)
  #positions_to_test = [617,618,619,620,622,623,624,636,637,638,639]
  #positions_to_test =  [637,638]
  #positions_to_test = [55,108,109,353,410,411,426,441,442]
  #positions_to_test = [656,657,675,676,854,855,859,860,861,902]

  path = '/home/fcodevil/Datasets/DRC/DRC_O6/SeqTrain/'

  speed_list = []
  screen = ScreenManager()

  image_queue = deque()

  actions_queue = deque()

  screen.start_screen([200,88],3,4)

  images= [np.array([200,88,3]),np.array([200,88,3]),np.array([200,88,3])]
  actions = [Control(),Control(),Control()]

  

  for h_num in positions_to_test:

    print " SEQUENCE NUMBER ",h_num
    data = h5py.File(path+'data_'+ str(h_num).zfill(5) +'.h5', "r+")

    #redata = h5py.File('/media/adas/012B4138528FF294/NewGTA/redata_'+ str(
    



    # skip to highway
    for i in range(0,198,12):




      #img = cam['X'][log['cam1_ptr'][i]].swapaxes(0,2).swapaxes(0,1)

      img_1 = np.array(data['images_center'][i]).astype(np.uint8)


      img_2 = np.array(data['images_center'][i+1]).astype(np.uint8)


      img_3 = np.array(data['images_center'][i+2]).astype(np.uint8)

      #rint int(data['targets'][i][49])

      

      
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



      direction = data['targets'][i][8]
      #actions[0].steer +=0.318
      #actions[1].steer +=0.318
      #actions[2].steer +=0.318

      print actions[1].steer 

      #print data['targets'][i][9]

      #print data['targets'][i][10]
      #print "Speed"
      #print data['targets'][i][18]



      #print "Mag"
      #print data['targets'][i][30]
      #compass = np.deg2rad(data['targets'][i][30])
     # print 'rad ',compass
      #print 'y proj ',np.sin(compass),'speed ',data['targets'][i][19]
      #print 'x proj ',np.cos(compass),'speed ',data['targets'][i][18]

      #print "Linear Speed"
      # car_speed = math.sqrt(data['targets'][i][18]*data['targets'][i][18] + data['targets'][i][19]*data['targets'][i][19])
      #print car_speed
      #print 'car_speed ',data['targets'][i][51]


      speed_list.append((actions[1].steer))

      #speed_list.append(((actions[1].steer +((1-0.318)))/(1+(1-0.318)))

      #print actions[1].steer
      #for j in range(3):
      #  screen.plot3camrc( 0,images[j],\
      #        actions[j],direction,0,\
      #        [data['targets'][i][42],data['targets'][i][43]],j) #


      #reimg = np.array(redata['images_center'][i])
      #recontrol_input = np.array(redata['control'][i][1])
      #print img
      #img = img*255
      #print img

  plt.plot(range(0,len(speed_list)),speed_list)
  
  plt.show()
  #save_gta_surface(gta_surface)




