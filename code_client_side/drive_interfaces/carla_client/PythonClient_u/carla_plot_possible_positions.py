from __future__ import print_function
from carla import CARLA

#from scene_parameters import SceneParams

import numpy as np

import matplotlib.pyplot as plt

import time


# instance a carla universe with the configurations provided inside the config class
carla =CARLA('127.0.0.1',2003)




positions_proto = carla.loadConfigurationFile('CarlaSettings.ini')

print ("RECEIVED POSIBLE POSITIONS")
positions =[]

for i in positions_proto:
 	positions.append([i.location.x,-i.location.y])


plt.scatter(*zip(*positions))    

labels = ['{0}'.format(i) for i in range(len(positions))]
for label, pos in zip(labels, positions):
    plt.annotate(
        label,
        xy=(pos[0], pos[1]), xytext=(0, 0),
        textcoords='offset points', ha='right', va='bottom',)

plt.show()





	
	
