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
import random



sys.path.append('drive_interfaces')

from screen_manager import ScreenManager

sensors = {'RGB':3,'labels':3,'depth':3}
resolution = [200,88]
camera_id_position = 25
direction_position = 24
speed_position = 10
number_of_seg_classes = 5
classes_join = {0:2,1:2,2:2,3:2,5:2,12:2,9:2,11:2,4:0,10:1,8:3,6:3,7:4}

#gta_surface = get_gta_map_surface()


# THis script has the following objectives

# * Convert the semantic segmentation format
# * Add random augmentation to the direction
# * Point critical points passible of mistakes


def join_classes(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in join_dic.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value


  return compressed_labels_image


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

  #positions_to_test = [93,104,170,173,229,245,283,397,413,425,565,581,591]
  #positions_to_test = range(0,660)
  #positions_to_test = [617,618,619,620,622,623,624,636,637,638,639]
  #positions_to_test =  [637,638]
  #positions_to_test = [55,108,109,353,410,411,426,441,442]
  #positions_to_test = [656,657,675,676,854,855,859,860,861,902]


  path = '/media/matthias/7E0CF8640CF818BB/Github/ModularEnd2End/Desktop/20171025_Weather8_1/'

  positions_to_test =  range(67,len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
  print positions_to_test


  path_clean = '/media/matthias/7E0CF8640CF818BB/Github/ModularEnd2End/Desktop/20171025_Weather8_1_clean/'


  if not os.path.exists(path_clean):
    os.mkdir( path_clean)


  for h_num in positions_to_test:

    print " SEQUENCE NUMBER ",h_num
    try:
      data = h5py.File(path+'data_'+ str(h_num).zfill(5) +'.h5', "r")
    except Exception as e:
      print e
      continue


    new_data = h5py.File(path_clean+'data_'+ str(h_num).zfill(5) +'.h5', "w")
    new_data_images= new_data.create_dataset('rgb', (200,88,200,3),dtype=np.uint8)
    new_data_labels= new_data.create_dataset('labels', (200,88,200,1),dtype=np.uint8)
    new_data_depth= new_data.create_dataset('depth', (200,88,200,3),dtype=np.uint8)
    targets  = new_data.create_dataset('targets', (200, data['targets'][0].shape[0]),'f')

    for i in range(0,200):

      new_data_images[i] = data['rgb'][i]

      new_data_labels[i] = join_classes(data['labels'][i],classes_join)*int(255/(number_of_seg_classes-1))
      
      new_data_depth[i] = data['depth'][i]


      target_array = np.zeros((data['targets'][0].shape[0]))

      target_array = data['targets'][i]


      if target_array[direction_position] == 2.0:
        multi_dim_coin = random.randint(0,9)
        if multi_dim_coin < 2:
          target_array[direction_position] =random.sample([3.0,4.0,5.0], 1)[0]

      if target_array[0] > 0.65 or target_array[0] < -0.65:
        print 'WARNING : WEIRD STEERING'
        os.remove(path_clean+'data_'+ str(h_num).zfill(5) +'.h5')
        break



      new_data['targets'][i] = target_array



      #recontrol_input = np.array(redata['control'][i][1])
      #print img
      #img = img*255
      #print img



  #save_gta_surface(gta_surface)




