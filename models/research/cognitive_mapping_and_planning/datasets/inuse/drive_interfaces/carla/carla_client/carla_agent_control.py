"""
    Sample Agent controlling for carla. Please refer to carla_use_example for a simpler and more
    documented example.

"""
from __future__ import print_function
from carla import CARLA


#from scene_parameters import SceneParams

from PIL import Image
import numpy as np

import os
import random
import time
import sys
import argparse
import logging
from socket import error as socket_error
import scipy
from carla import Waypointer

from carla import Control,Measurements
from carla.agent import *
import math
#from noiser import Noiser


import pygame
from pygame.locals import *


sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

def test_reach_goal(player,goal):
    distance = sldist([player.location.x,player.location.y],[goal.location.x,goal.location.y])
    if distance < 300.0:
        return True
    else:
        return False

#poses =[[19,66],[79,14],[19,57],[23,1],\
#      [53,76],[42,13],[31,71],[33,5],\
#      [54,30],[10,61],[66,3],[27,12],\
#      [79,19],[2,29],[16,14],[5,57],\
#      [70,73],[46,67],[57,50],[61,49],[21,12],\
#      [51,81],[77,68],[56,65],[43,54]]
#poses =#[[132,27],[132,27],[132,27]]   #[105,29],[27,130],[102,87],#[132,27],[24,44],\
#          [96,26],[34,67],[28,1],[140,134],[105,9],\
#           [148,129],[21,16],[147,97],[42,51],\
#          [30,41],[18,107],[69,45],[102,95],\
#           [111,64],[79,45],[84,69],[37,81]]

def find_valid_episode_position(positions,waypointer):
    found_match = False
    while not found_match:
        index_start = np.random.randint(len(positions))
        start_pos =positions[index_start]
        if not waypointer.test_position((start_pos.location.x,start_pos.location.y,22),\
            (start_pos.orientation.x,start_pos.orientation.y,start_pos.orientation.z)):
            continue
        index_goal = np.random.randint(len(positions))
        if index_goal == index_start:
            continue


        print (' TESTING (',index_start,',',index_goal,')')
        goals_pos =positions[index_goal]  
        if not waypointer.test_position((goals_pos.location.x,goals_pos.location.y,22),\
            (goals_pos.orientation.x,goals_pos.orientation.y,goals_pos.orientation.z)):
            continue
        if sldist([start_pos.location.x,start_pos.location.y],[goals_pos.location.x,goals_pos.location.y]) < 25000.0:
            print ('COntinued on distance ', sldist([start_pos.location.x,start_pos.location.y],[goals_pos.location.x,goals_pos.location.y]))
            
            continue

        if waypointer.test_pair((start_pos.location.x,start_pos.location.y,22)\
            ,(start_pos.orientation.x,start_pos.orientation.y,start_pos.orientation.z),\
            (goals_pos.location.x,goals_pos.location.y,22)):
            found_match=True
        waypointer.reset()

    return 21,40



class App:
    def __init__(self, port=2000, host='vcl-gpu1', config='./CarlaSettings.ini',\
    resolution=(1650,1950), plot_depths=False,verbose=True,plot_map=True):
    
        self._running = True
        self._display_surf = None
        self.port = port
        self.host = host
        self.ini = config
        self.verbose = verbose
        self.resolution = resolution
        self.size = self.weight, self.height = resolution
        self.plot_depths = plot_depths
        self.plot_map = plot_map
        self.config = ConfigAgent('carla_1')
        #self.noiser = Noiser('Spike')
        if plot_map:

            self.agent = Agent(self.config)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        logging.debug('Started the PyGame Library')
        self._running = True
        self.step = 0
        self.prev_step = 0
        self.prev_time = time.time()
        

        #print (config.port)
        self.carla =CARLA(self.host, self.port)

        self.positions = self.carla.loadConfigurationFile(self.ini)
        #all_positions =set()
        #for i in range(10000):
            #print ('Iter ',i)

        self.index_start,self.index_goal=find_valid_episode_position(self.positions,self.agent.waypointer)
            #all_positions.add((self.index_start,self.index_goal))
            
        #episode_config =poses[np.random.randint(len(poses))]

        print (' Found (',self.index_start,',',self.index_goal,')') 

        #print (all_positions)
        #print ('FOUND ',len(all_positions))
        #exit()
        #print ("Episode on POS ",episode_config[0])

        #self.goals_pos = self.positions[episode_config[1]]
        #print ('GOAL PoS ',episode_config[1])

        #self.carla.newEpisode(episode_config[0])
        self.carla.newEpisode(self.index_start)

        self.prev_restart_time = time.time()
 
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
    def on_loop(self):
        self.step += 1

        keys=pygame.key.get_pressed()
        measurements = self.carla.getMeasurements()
        restart = False
        if keys[K_r]:
            pressed_keys.append('r')
            if time.time() - self.prev_restart_time > 2.:
                self.prev_restart_time = time.time()
                restart = True
        pack = measurements['PlayerMeasurements']
        self.img_vec = measurements['BGRA']
 	#seg = measurements['SEG']
        #self.depth_vec = data[2]
        print ('Start,GOAL ',self.index_start,self.index_goal)
        control=self.agent.get_control(measurements,self.positions[self.index_goal])
        if test_reach_goal(pack.transform,self.positions[self.index_goal]):

            self.agent.waypointer.reset()
            self.index_start,self.index_goal =  find_valid_episode_position(self.positions,self.agent.waypointer)

            self.carla.newEpisode(self.index_start)


        #control,_,_ = self.noiser.compute_noise(control,measurements['PlayerMeasurements'].forward_speed)
        #print pack.depth1
        self.carla.sendCommand(control)
        #time.sleep(0.2)

        if time.time() - self.prev_time > 1.:
            print('Step', self.step, 'FPS', float(self.step - self.prev_step) / (time.time() - self.prev_time))

            print('speed', pack.forward_speed, 'collision', pack.collision_other, \
                'collision_car', pack.collision_vehicles, 'colision_ped', pack.collision_pedestrians)
            self.prev_step = self.step
            self.prev_time = time.time()
            
        if restart:
            print('\n *** RESTART *** \n')
    
            player_pos = np.random.randint(self.num_pos)


            print('  Player pos %d' % (player_pos))
            self.carla.newEpisode(player_pos)


        
        
    def on_render(self):

        pos_x =0

        #for i in range(len(self.img_vec)):
        #    self.img_vec[i] = self.img_vec[i][:,:,:3]
        #    self.img_vec[i] = self.img_vec[i][:,:,::-1]

            #print (self.img_vec[i])
        #    surface = pygame.surfarray.make_surface(np.transpose(self.img_vec[i], (1,0,2)))
        #    self._display_surf.blit(surface,(pos_x,0))
        #    pos_x += self.img_vec[i].shape[1]

        pos_x =0
        if self.plot_map:
            #print (self.waypointer.search_image)
            search_image =self.agent.waypointer.search_image[:,:,:3]
            search_image= scipy.misc.imresize(search_image,[1950,1650])

            surface = pygame.surfarray.make_surface(np.transpose(search_image, (1,0,2)))
            self._display_surf.blit(surface,(pos_x,0))



        if len(self.img_vec)> 0:

            pos_y =self.img_vec[0].shape[0]


        if self.plot_depths:
            for i in range(len(self.depth_vec)):
                surface = pygame.surfarray.make_surface(np.transpose(self.depth_vec[i], (1,0,2)))
                self._display_surf.blit(surface,(pos_x,pos_y))
                pos_x += self.depth_vec[i].shape[1] 


            
        pygame.display.flip()

        
    def on_cleanup(self):
        self.carla.close_conections()
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            #try:

            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()

            #except Exception:
                #   self._running = False
                #break


        self.on_cleanup()
            
                

        
 
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Run multiple servers on multiple GPUs')
    parser.add_argument('host', metavar='HOST', type=str, help='host to connect to')
    parser.add_argument('port', metavar='PORT', type=int, help='port to connect to')
	

    parser.add_argument("-c", "--config", help="the path for the server config file that the client sends",type=str,default="./CarlaSettings.ini") 

    parser.add_argument("-pd", "--plot_depths", help="activate the log file",action="store_true")


    parser.add_argument("-l", "--log", help="activate the log file",action="store_true") 
    parser.add_argument("-lv", "--log_verbose", help="put the log file to screen",action="store_true") 
    args = parser.parse_args()

    print(args)

    if args.log or args.log_verbose:
        LOG_FILENAME = 'log_manual_control.log'
        logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
        if args.log_verbose:  # set of functions to put the logging to screen


            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            root.addHandler(ch)
        


    theApp = App(port=args.port, host=args.host, config=args.config)
    theApp.on_execute()
    
    
    
    
