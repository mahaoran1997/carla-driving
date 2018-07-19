"""
    Keyboard controlling for carla. Please refer to carla_use_example for a simpler and more
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

from carla import  Control,Measurements

import pygame
from pygame.locals import *


class App:
    def __init__(self, port=2000, host='vcl-gpu1', config='./CarlaSettings.ini',\
    resolution=(1400,1400), plot_depths=False,verbose=True,plot_map=True):
    
        self._running = True
        self._display_surf = None
        self.port = port
        self.host = host
        self.config = config
        self.verbose = verbose
        self.resolution = resolution
        self.size = self.weight, self.height = resolution
        self.plot_depths = plot_depths
        self.plot_map = plot_map
        if plot_map:
            city_name ='carla_0'
            dir_path = os.path.dirname(__file__)
            self.waypointer = Waypointer(dir_path+'carla/planner/' + city_name + '.txt',\
                dir_path+'carla/planner/' + city_name + '.png')

 
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

        positions = self.carla.loadConfigurationFile(self.config)
        self.num_pos = len(positions)
        episode_start =np.random.randint(self.num_pos)
        print ("Episode on POS ",episode_start)
        goal_pos =np.random.randint(self.num_pos)
        self.goals_pos = positions[goal_pos]
        print ('GOAL PoS ',goal_pos)

        self.carla.newEpisode(episode_start)


        self.prev_restart_time = time.time()
 
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
    def on_loop(self):
        self.step += 1
        keys=pygame.key.get_pressed()
        throttle = 0
        steer = 0
        restart = False
        pressed_keys = []
        if keys[K_LEFT]:
            steer = -1.
            pressed_keys.append('left')
        if keys[K_RIGHT]:
            pressed_keys.append('right')
            steer = 1.
        if keys[K_UP]:
            pressed_keys.append('up')
            throttle = 1.
        if keys[K_DOWN]:
            pressed_keys.append('down')
            throttle = -1.
        if keys[K_r]:
            pressed_keys.append('r')
            if time.time() - self.prev_restart_time > 2.:
                self.prev_restart_time = time.time()
                restart = True
        if time.time() - self.prev_restart_time < 2.:
            throttle = 0.
            steer = 0.

    	control = Control()
    	control.throttle = throttle
    	control.steer = steer
        self.carla.sendCommand(control)
        measurements = self.carla.getMeasurements()
        pack = measurements['PlayerMeasurements']
        self.img_vec = measurements['BGRA']
        #self.depth_vec = data[2]

        points = self.waypointer.get_next_waypoints((pack.transform.location.x,pack.transform.location.y,22)\
            ,(pack.transform.orientation.x,pack.transform.orientation.y,pack.transform.orientation.z)\
            ,(self.goals_pos.location.x,self.goals_pos.location.y,self.goals_pos.location.z)\
            ,(1,0,0))


        #print (points)
        #print pack.depth1

        
        if time.time() - self.prev_time > 1.:
            print('Step', self.step, 'FPS', float(self.step - self.prev_step) / (time.time() - self.prev_time))

            print('speed', pack.forward_speed, 'collision', pack.collision_other, \
                'collision_car', pack.collision_vehicles, 'colision_ped', pack.collision_pedestrians, 'pressed:', pressed_keys)
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
            search_image =self.waypointer.search_image[:,:,:3]
            #search_image= scipy.misc.imresize(search_image,[400,300])

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
    
    
    
    
