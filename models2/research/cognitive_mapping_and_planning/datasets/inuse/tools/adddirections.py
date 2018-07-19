#!/usr/bin/env python
import sys

sys.path.append('utils')
sys.path.append('configuration')

import glob
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


def string_to_node(string):
  vec = string.split(',')

  return (int(vec[0]),int(vec[1]))

def string_to_floats(string):
  vec = string.split(',')

  return (float(vec[0]),float(vec[1]),float(vec[2]))


with  open('/home/fcodevil/chauffeur/drive_interfaces/carla_interface/carla_1.txt', 'r') as file:

  linewordloffset = file.readline()
  # The offset of the world from the zero coordinates ( The coordinate we consider zero)
  worldoffset = string_to_floats(linewordloffset)

  #WARNING: for now just considering the y angle
  lineworldangles = file.readline()
  angles =  string_to_floats(lineworldangles)
  #self.worldrotation = np.array([[math.cos(math.radians(self.angles[0])),0,math.sin(math.radians(self.angles[0])) ],[0,1,0],[-math.sin(math.radians(self.angles[0])),0,math.cos(math.radians(self.angles[0]))]])

  worldrotation = np.array([[math.cos(math.radians(angles[2])),-math.sin(math.radians(angles[2])) ,0.0],[math.sin(math.radians(angles[2])),math.cos(math.radians(angles[2])),0.0],[0.0,0.0,1.0]])

  # Ignore for now
  lineworscale = file.readline()

  linemapoffset = file.readline()

  # The offset of the map zero coordinate
  mapoffset =  string_to_floats(linemapoffset)   

  # the graph resolution.
  linegraphres = file.readline()
  resolution =  string_to_node(linegraphres) 

map_image = Image.open('/home/fcodevil/chauffeur/drive_interfaces/carla_interface/carla_1p.png')
map_image.load()
map_image = np.asarray(map_image, dtype="int32" )
# The number of game units per pixel 
pixel_density = 16.43
#A pixel positions with respect to graph node position is:  Pixel = Node*50 +2
node_density = 50.0

def get_target_ori(target_pos):




  relative_location = []
  pixel=[]
  rotation = np.array([target_pos[0],target_pos[1],target_pos[2]])
  rotation = rotation.dot(worldrotation)

  #print 'rot ', rotation
  
  relative_location.append(rotation[0] + worldoffset[0] - mapoffset[0])
  relative_location.append(rotation[1] + worldoffset[1] - mapoffset[1])
  relative_location.append(rotation[2] + worldoffset[2] - mapoffset[2])
  #print 'trans ', relative_location

  pixel.append(math.floor(relative_location[0]/float(pixel_density)))
  pixel.append(math.floor(relative_location[1]/float(pixel_density)))
  #print self.map_image.shape
  ori = map_image[int(pixel[1]),int(pixel[0]),2]
  ori = ((float(ori)/255.0) ) *2*math.pi 

  #print self.map_image[int(pixel[1]),int(pixel[0]),:]
  #print ori
  #print (math.cos(ori),math.sin(ori))
  #print exit()

  return ori

def read_all_files(file_names):
 


  dataset_names = ['targets']
  datasets_cat = [list([]) for _ in xrange(len(dataset_names))]



  lastidx = 0
  count =0
  #print file_names
  for cword in file_names:
    try:
        print cword
        print count
        dset = h5py.File(cword, "r")  



        for i in range(len(dataset_names)):

          dset_to_append = dset[dataset_names[i]]


          #if dset_to_append.shape[1] <23:
          #  zero_vec = np.zeros((1,dset_to_append.shape[0]))

          #  dset_to_append = np.insert(dset_to_append,15,zero_vec,axis=1)

          if dset_to_append.shape[1] >23 and dset_to_append.shape[1] <28:  # carla case
            zero_vec = np.zeros((dset_to_append.shape[0],1))

            dset_to_append = np.append(dset_to_append,zero_vec,axis=1)


          datasets_cat[i].append( dset_to_append[:])



        
        dset.flush()
        count +=1

    except IOError:
      import traceback
      exc_type, exc_value, exc_traceback  = sys.exc_info()
      traceback.print_exc()
      traceback.print_tb(exc_traceback,limit=1, file=sys.stdout)
      traceback.print_exception(exc_type, exc_value, exc_traceback,
                            limit=2, file=sys.stdout)
      print "failed to open", cword




  for i in range(len(dataset_names)):     
    datasets_cat[i] = np.concatenate(datasets_cat[i], axis=0)
    datasets_cat[i] = datasets_cat[i].transpose((1,0))



  return datasets_cat


import random



# ***** main loop *****
if __name__ == "__main__":
  
  # Concatenate all files
  path = '/home/fcodevil/Datasets/CarlaNew/CarlaAgentTest/SeqTrain/'
  path_dir = '/home/fcodevil/Datasets/CarlaNew/CarlaAgentTest_dir/'
  files = [os.path.join(path, f) for f in glob.glob1(path, "data_*.h5")]

  targets = read_all_files(sorted(files))[0]
  # Compute all angles



  print targets.shape[1]
  angles =[]
  for i in range(targets.shape[1]):
    angles.append(get_target_ori((targets[8][i],targets[9][i],22)))

  #print angles
  angles = np.array(angles)
  print angles.shape
  intersections = np.zeros(angles.shape)
  # Get all indexes where  angle is equal to 6.28
  intersections[np.where(angles>=6.2)] = 1.0

  # Get the border indexes

  border = intersections[:-1] != intersections[1:]
  borderIndexes = np.where(border==True)
  print borderIndexes[0].shape[0]

  direction = np.ones(angles.shape)*2.0
  # Go n points after and n points before and get the medium to detect the direction
  last_end =0
  for i in range(0,borderIndexes[0].shape[0],2):
    init = borderIndexes[0][i]
    end = borderIndexes[0][i+1]



    print 'angle init', angles[init-1]
    print 'angle end', angles[end+1]
    difference = angles[init-1] - angles[end+1]
    if (difference > 0 and difference < 3.0) or difference <-3.0: 
      inter = 3.0
    elif difference < 0 or difference > 3.0:
      inter =4.0
    else:
      inter=5.0
    print inter

    print 'init ',init
    print 'end ',end
    
    N = min(100,init-1-last_end) + random.randint(0,max(1,init-1-last_end -100))
    direction[((init-1)-N):((end+1)+10)] = inter

    last_end = end

  #plt.plot(range(0,5000),angles[0:5000],'g')#range(0,len(speed_list)),speed_list_noise,'b',range(0,len(speed_list)),just_noise,'r')
  
  #plt.show()

  # Apply the direction to N points before the border index

  # Now go again over all files and change the direction   
  sequence_num = range(0,4788)
  for h_num in sequence_num:
    try:
      print " SEQUENCE NUMBER ",h_num
      data = h5py.File(path+'data_'+ str(h_num).zfill(5) +'.h5', "r")

      #redata = h5py.File('/media/adas/012B
      new_data = h5py.File(path_dir+'data_'+ str(h_num).zfill(5) +'.h5', "w")
      new_data_center= new_data.create_dataset('images_center', (200,88,200,3),dtype=np.uint8)
      targets  = new_data.create_dataset('targets', (200, data['targets'][0].shape[0]),'f')



      for i in range(0,200):

        new_data_center[i] = data['images_center'][i]
        target_array = np.zeros((26))

        target_array = data['targets'][i]

        target_array[24] = direction[i + 200*h_num]



        new_data['targets'][i] = target_array
    except:
      continue



  plt.plot(range(0,direction.shape[0]),direction,'g')#range(0,len(speed_list)),speed_list_noise,'b',range(0,len(speed_list)),just_noise,'r')
  
  plt.show()
  #save_gta_surface(gta_surface)




