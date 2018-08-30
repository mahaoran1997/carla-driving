import sys
import argparse
import os
import logging
import traceback
import math
import configparser
import datetime
import numpy as np
import time
import sys
import socket
import logging

import copy

import h5py
import scipy
from PIL import Image

from ConfigParser import ConfigParser
from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from contextlib import contextmanager


import src.utils as utils
'''import re
import matplotlib.pyplot as plt

import graph_tool as gt
import graph_tool.topology

from tensorflow.python.platform import gfile
import logging
import src.file_utils as fu

import src.graph_utils as gu
import src.map_utils as mu
import src.depth_utils as du
import render.swiftshader_renderer as sru
from render.swiftshader_renderer import SwiftshaderRenderer
import cv2'''

#TODO: action_noise does not use rng


from datasets.inuse.drive_interfaces.driver import Driver
from datasets.inuse.drive_interfaces.configuration.cognitive_drive_config import configDrive
from datasets.inuse.drive_interfaces.noiser import Noiser
# from datasets.inuse.drive_interfaces.carla.comercial_cars.carla_human import CarlaHuman
from datasets.inuse.drive_interfaces.carla.carla_client.carla.carla import CARLA, Measurements, Control
from datasets.inuse.drive_interfaces.carla.carla_client.carla.agent import *
from datasets.inuse.drive_interfaces.carla.carla_client.carla import Planner


'''
from datasets.inuse.drive_interfaces.carla.carla_client.benchmark_client.carla.benchmarks.agent import Agent as AgentBench
from datasets.inuse.drive_interfaces.carla.carla_client.benchmark_client.carla.benchmarks.corl_2017 import CoRL2017

from datasets.inuse.drive_interfaces.carla.carla_client.benchmark_client.carla.client import make_carla_client, VehicleControl
from datasets.inuse.drive_interfaces.carla.carla_client.benchmark_client.carla.tcp import TCPConnectionError
'''
Training = True #False
batch_size = 1
if Training:
    batch_size = 4
    cityfile = 'datasets/inuse/drive_interfaces/carla/carla_client/carla/planner/carla_1.png'
else:
    cityfile= 'datasets/inuse/drive_interfaces/carla/carla_client/carla/planner/carla_1.png'
readout_img = Image.open(cityfile)
img_array = np.asarray(readout_img)

readout = False

saveimgs = False



sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class ImageData():

    def __init__(self):
        self.raw_rgb = []
        self.rgb = []
        self.depth = []
        self.scene_seg = []


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))


def get_camera_dict(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    print (ini_file)
    print (config)
    cameras = config['CARLA/SceneCapture']['Cameras']
    camera_dict = {}
    cameras = cameras.split(',')
    print cameras
    for i in range(len(cameras)):
        angle = config['CARLA/SceneCapture/' + cameras[i]]['CameraRotationYaw']
        camera_dict.update({i: (cameras[i], angle)})
    return camera_dict

def compute_angle(x0, y0, x1, y1):
    cosv = (x0*x1 + y0*y1)/np.sqrt((x0**2+y0**2)*(x1**2+y1**2))
    #print x0, y0, x1, y1, (x0**2+y0**2)*(x1**2+y1**2), cosv
    return cosv

def find_valid_episode_position(positions, waypointer, rng, difficulty):

    debugging = False

    found_match = False
    while not found_match:
        index_start = rng.choice(range(len(positions)))
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
    waypointer.reset()
    return index_start, index_goal


class CarlaEnvMultiplexer:
    def __init__(self, logdir):
        logdir += '/pics'
        if saveimgs and (not os.path.exists(logdir)):
            os.makedirs(logdir)
        #self.drive_config = configDrive()
        self.drive_configs = []
        for i in range(batch_size):
            self.drive_configs.append(configDrive())#, self.drive_config, self.drive_config, self.drive_config]
        # TODO: adding more carla_configs
        prefix = './datasets/inuse/drive_interfaces/carla/CarlaSettingCity1W'
        self.carla_configs = []
        for i in range(15):
            self.carla_configs.append(prefix+str(i)+'.ini')
        self.carla_env_wrapper = None
        #self.driver = self.get_instance()
        self.logdir = logdir

    def sample_env(self, rngs):
        #get carla_env_wrapper
        rng = rngs[0]
        for i in range(batch_size):
            self.drive_configs[i].port = 2000 + i * 3
            self.drive_configs[i].carla_config = rng.choice(self.carla_configs)
        for i in range(batch_size):
            print (self.drive_configs[i].port)
        if self.carla_env_wrapper != None:
            print ("reset_env!!!!!!!!!")
            self.carla_env_wrapper.reset_config(self.drive_configs)
        else:
            self.carla_env_wrapper = CarlaEnvWrapper(self.drive_configs, self.logdir)
        return self.carla_env_wrapper
    
    def close(self):
        self.carla_env_wrapper.close()



class CarlaEnvWrapper():

    def __init__(self, drive_configs, logdir):
        #create_env
        self.need_reset = False
        self.drive_configs = drive_configs
        self.carla_envs = []
        self.history = []
        self.logdir = logdir
        for i in range(len(drive_configs)):
            print ('Environment {:d}'.format(i))
            self.carla_envs.append(self.create_env(drive_configs[i], i))
            self.history.append([])
        
    def reset_config(self, drive_configs):
        self.need_reset = True
        self.drive_configs = drive_configs
        self.history = []
        for i in range(len(drive_configs)):
            print ('Environment {:d}'.format(i))
            self.carla_envs[i].reset_config(drive_configs[i])
            self.history.append([])
            camera_dict = get_camera_dict(self.drive_configs[i].carla_config)
            print " Camera Dict "
            print camera_dict

    def close(self):
        #self.carla_envs
        for i in range(len(self.carla_envs)):
            self.carla_envs[i].close()

    def reset(self, rngs):
        #restart every carla_env
        start_indices = []
        rng = rngs[0]
        self.history = []
        for i in range(len(self.carla_envs)):
            print ('Environment {:d} start'.format(i))
            if not self.need_reset:
                start_index = self.carla_envs[i].start(rng)
            else:
                print ('reset{:d}'.format(i))
                start_index = self.carla_envs[i].reset(rng)
            start_indices.append(start_index)
            self.history.append([])
        return start_indices

    def create_env(self, drive_config, env_id):
        #get carla_env but do not start
        driver = CarlaEnv(drive_config, env_id, self.logdir)
        camera_dict = get_camera_dict(drive_config.carla_config)
        print " Camera Dict "
        print camera_dict
        # folder_name = str(datetime.datetime.today().year) + str(datetime.datetime.today().month) + str(datetime.datetime.today().day)
        return driver

    def get_common_data(self):

        #mhr:!!!!!!!note that here line and map are references
        '''maps = [[],[],[],[]]
        line = []
        for i in range(524):
            line.append([0.0])
        map = []
        for i in range(1112):
            map.append(line)
        for i in range(4):
            maps[i].append(map)'''
        maps = np.zeros((batch_size,1,1,1,1)) #((4,1,1112,524,1))
        rel_goal_locs = []
        goal_locs = []
        #goal_locs.append([])
        for i in range(len(self.carla_envs)):
            rel_goal_loc, goal_loc = self.carla_envs[i].get_goal_loc_at_start()
            rel_goal_locs.append([rel_goal_loc])
            goal_locs.append([goal_loc])
        maps = np.array(maps).astype(np.float32)
        goal_locs = (np.array(goal_locs)).astype(np.float32)
        rel_goal_locs = (np.array(rel_goal_locs)).astype(np.float32)
        return vars(utils.Foo(orig_maps=maps, goal_loc=goal_locs, rel_goal_loc_at_start=rel_goal_locs))

    def image_preprocess(self, image):
        if self.drive_configs[0].typ == 'rgb':
            image = image.astype(np.float32) * 1.0 - 128
        else:
            #TODO: really need to do this?
            depth = image.astype(np.float32)
               # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            image = np.dot(depth[:, :, :3], [65536.0, 256.0, 1.0])
            image /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
            image *= 8.0
            d = image[...][...,np.newaxis]*1.
            d[d < 0.01] = np.NaN
            isnan = np.isnan(d)
            d = 100./d
            d[isnan] = 0.
            image = np.concatenate((d, isnan), axis=d.ndim-1)
        return image

    def get_features_name(self):
        f = []
        f.append('imgs')
        #f.append('rel_goal_loc')
        f.append('loc_on_map')
        f.append('gt_dist_to_goal')
        if readout:
            for i in range(len(self.drive_configs[0].map_scales)):
                f.append('readout_maps_{:d}'.format(i))
        #for i in range(len(self.drive_configs[0].map_scales)):
        #    f.append('ego_goal_imgs_{:d}'.format(i))
        f.append('speed')
        f.append('command')
        f.append('incremental_locs')
        f.append('incremental_thetas')
        f.append('node_ids')
        f.append('perturbs')
        f.append('measurements')
        return f

    def get_features(self, current_node_ids, step_number):
        '''loc_on_map
            ego_goal_imgs_0,1,2
            node_ids
            'incremental_thetas'
            'gt_dist_to_goal'
            'imgs'
            'perturbs'
            
            'incremental_locs'
        '''
        outs = {}
        outs['loc_on_map'] = []
        #outs['ego_goal_imgs_0'] = []
        #outs['ego_goal_imgs_1'] = []
        #outs['ego_goal_imgs_2'] = []
        outs['incremental_thetas'] = []
        outs['imgs'] = []
        outs['incremental_locs'] = []
        outs['measurements'] = []
        outs['command'] = []
        outs['speed'] = []

        if readout:
            outs['readout_maps_0'] = []
            outs['readout_maps_1'] = []
            outs['readout_maps_2'] = []

        #useless inputs
        outs['node_ids'] = []
        outs['gt_dist_to_goal'] = []
        outs['perturbs'] = []
        oris = []
        
        
        
        for i in range(len(self.carla_envs)):
            if(step_number == 0):
                self.carla_envs[i].skip_frames()
                image, measurements, direction, reach_goal, action_noisy, control = self.carla_envs[i].get_data_for_mapper(step_number)
                self.history[i].append((image, measurements, direction, reach_goal, action_noisy, control))
            else:
                image, measurements, direction, reach_goal, action_noisy, control = self.history[i][step_number]
            outs['loc_on_map'].append([[measurements['PlayerMeasurements'].transform.location.x, measurements['PlayerMeasurements'].transform.location.y]])
            '''goal_imgs = self.carla_envs[i].get_ego_goal_img()
            for j in range(len(goal_imgs)):
                outs['ego_goal_imgs_{:d}'.format(j)].append(goal_imgs[j])'''
            if readout:
                readout_maps = self.carla_envs[i].get_readout_maps(measurements['PlayerMeasurements'].transform)
                for j in range(len(readout_maps)):
                    outs['readout_maps_{:d}'.format(j)].append(readout_maps[j])
            current_theta = np.arctan2(measurements['PlayerMeasurements'].transform.orientation.y, measurements['PlayerMeasurements'].transform.orientation.x)
            if (step_number > 0):
                if measurements['PlayerMeasurements'].transform.location.y-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.y < 0.1 and measurements['PlayerMeasurements'].transform.location.x-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.x < 0.1:
                    translation_theta = current_theta
                else:
                    translation_theta = np.arctan2(measurements['PlayerMeasurements'].transform.location.y-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.y, measurements['PlayerMeasurements'].transform.location.x-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.x)
                previous_theta = np.arctan2(self.history[i][step_number-1][1]['PlayerMeasurements'].transform.orientation.y, self.history[i][step_number-1][1]['PlayerMeasurements'].transform.orientation.x)
            else:
                translation_theta = current_theta
                previous_theta = current_theta
            outs['incremental_thetas'].append([[-(translation_theta - previous_theta), -(current_theta - translation_theta)]])
            outs['imgs'].append([[self.image_preprocess(image)]])
            if step_number > 0:
                square = np.square(measurements['PlayerMeasurements'].transform.location.y-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.y) +np.square(measurements['PlayerMeasurements'].transform.location.x-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.x)
                length = np.sqrt(square)
            else:
                length = 0.0
            outs['incremental_locs'].append([[0.0, length]])
            ax, ay = measurements['PlayerMeasurements'].acceleration.x, measurements['PlayerMeasurements'].acceleration.y
            
            ay = 0 - ay
            current_pos = measurements['PlayerMeasurements'].transform
            orientation_vec_square = np.square(current_pos.orientation.x)+np.square(current_pos.orientation.y)
            orientation_vec_length = np.sqrt(orientation_vec_square)
            #goal_orientation_vec_square = np.square(goal_pos.orientation.x)+np.square(goal_pos.orientation.y)
            #goal_orientation_vec_length = np.sqrt(goal_orientation_vec_square)
            acx = ((-current_pos.orientation.y)*ax-current_pos.orientation.x*ay)/orientation_vec_length
            acy = (current_pos.orientation.x*ax+(-current_pos.orientation.y)*ay)/orientation_vec_length
            outs['measurements'].append([[measurements['PlayerMeasurements'].forward_speed, acx, acy]])
            outs['speed'].append([[measurements['PlayerMeasurements'].forward_speed]])
            #directions ( 3 is left, 4 is right, 5 is straight)
            #direction -= 2.0
            outs['command'].append([[1.0, 0.0, 0.0, 0.0]])
            outs['command'][i][0][int(round(direction-2.0))] = 1.0
            #useless inputs
            outs['node_ids'].append([[self.carla_envs[i].episode_config[0]]]) 
            outs['gt_dist_to_goal'].append([[0.0]]) 
            outs['perturbs'].append([[0.0, 0.0, 0.0, 0.0]]) 

            oris.append([measurements['PlayerMeasurements'].transform.orientation.x, measurements['PlayerMeasurements'].transform.orientation.y])

        for key in outs.keys():
            outs[key] = np.array(outs[key]).astype(np.float32)
        return outs, oris

    


    
    def get_optimal_action(self, current_node_ids, step_number):
        """Returns the optimal action from the current node."""
        optimal_actions = []
        for i in range(len(self.carla_envs)):
            optimal_actions.append([self.history[i][step_number][4].steer, self.history[i][step_number][4].throttle, self.history[i][step_number][4].brake])
        optimal_actions = np.array(optimal_actions).astype(np.float32)
        return optimal_actions

    

    def get_targets(self, current_node_ids, step_number):
        """Returns the target actions from the current node."""
        optimal_actions = []
        for i in range(len(self.carla_envs)):
            optimal_actions.append([self.history[i][step_number][5].steer, self.history[i][step_number][5].throttle, self.history[i][step_number][5].brake])
        optimal_actions = np.array(optimal_actions).astype(np.float32)
        action = np.expand_dims(optimal_actions, axis=1)
        return vars(utils.Foo(action=action))
    
    def get_targets_name(self):
        """Returns the list of names of the targets."""
        return ['action']

    def take_action(self, current_node_ids, action, step_number):
        """In addition to returning the action, also returns the reward that the
        agent receives."""
        #mhr: TODO:improve performance by reducing the times to get sensor data
        
        #return starting node id
        starting_indices = [self.carla_envs]
        rewards = []

        for i in range(len(self.carla_envs)):
            self.carla_envs[i].act(action[i])
        
        for i in range(len(self.carla_envs)):
            img, measurements, direction, reach_goal, action_noisy, control = self.carla_envs[i].get_data_for_mapper(step_number)
            self.history[i].append((img, measurements, direction, reach_goal, action_noisy, control))
            reward = 0
            if reach_goal:
                reward = self.drive_configs[i].reward_at_goal
            reward -= self.drive_configs[i].reward_time_penalty
            rewards.append(reward)
            starting_indices.append(self.carla_envs[i].episode_config[0])
        return starting_indices, rewards
        
    


class CarlaEnv(Driver):

    def __init__(self, driver_conf, env_id, logdir):
        Driver.__init__(self)
        self.iter = 0
        self.logdir = logdir
        self.basedir = logdir
        self.reset_config(driver_conf)
        self.id = env_id
        
        
        # self._straight_button = False
        # self._left_button = False
        # self._right_button = False
        # self._recording= False

        #self._skiped_frames = 20  # ??? TODO: delete it

        # load a manager to deal with test data
        # self.use_planner = driver_conf.use_planner

        # if driver_conf.use_planner:
        
        '''self.planner = Planner('./datasets/inuse/drive_interfaces/carla/comercial_cars/' + driver_conf.city_name +
                               '.txt', './datasets/inuse/drive_interfaces/carla/comercial_cars/' + driver_conf.city_name + '.png')

        self._host = driver_conf.host
        self._port = driver_conf.port
        self._config_path = driver_conf.carla_config
        self._resolution = driver_conf.resolution
        self._image_cut = driver_conf.image_cut
        #self._autopilot = driver_conf.autopilot
        self._reset_period = driver_conf.reset_period
        self._driver_conf = driver_conf
        self.typ = driver_conf.typ
        self._rear = False
        #if self._autopilot:
        #print driver_conf
        self._agent = Agent(ConfigAgent(driver_conf.city_name, driver_conf.stop4TL))
        self.noiser = Noiser(driver_conf.noise)
        self._map_scales = driver_conf.map_scales
        self._map_crop_sizes = driver_conf.map_crop_sizes
        self._n_ori = driver_conf.n_ori
        self._stop4TL = driver_conf.stop4TL
        self._difficulty = driver_conf.difficulty
        #self._resolution = driver_conf.resolution
        self._dist_to_activate = driver_conf.dist_to_activate # TODO:?
        # Set 10 frames to skip
        self._skiped_frames = driver_conf.skiped_frames  # TODO:?'''

    def  reset_config(self, driver_conf):
        #self._skiped_frames = 20  # ??? TODO: delete it
        # if driver_conf.use_planner:
        self.planner = Planner('./datasets/inuse/drive_interfaces/carla/comercial_cars/' + driver_conf.city_name +
                               '.txt', './datasets/inuse/drive_interfaces/carla/comercial_cars/' + driver_conf.city_name + '.png')

        self._host = driver_conf.host
        self._port = driver_conf.port
        self._config_path = driver_conf.carla_config
        self._resolution = driver_conf.resolution
        self._image_cut = driver_conf.image_cut
        #self._autopilot = driver_conf.autopilot
        self._reset_period = driver_conf.reset_period
        self._driver_conf = driver_conf
        self.typ = driver_conf.typ
        self._rear = False
        #if self._autopilot:
        #print driver_conf
        self._agent = Agent(ConfigAgent(driver_conf.city_name, driver_conf.stop4TL))
        self.noiser = Noiser(driver_conf.noise)
        self._map_scales = driver_conf.map_scales
        self._map_crop_sizes = driver_conf.map_crop_sizes
        self._n_ori = driver_conf.n_ori
        self._stop4TL = driver_conf.stop4TL
        self._difficulty = driver_conf.difficulty
        #self._resolution = driver_conf.resolution
        self._dist_to_activate = driver_conf.dist_to_activate # TODO:?
        # Set 10 frames to skip
        self._skiped_frames = driver_conf.skiped_frames  # TODO:?
        self._replay_action = driver_conf.replay_action
        self.reach_goal = False
        self.iter += 1
        if Training:
            self.logdir = self.basedir + '/train/'+str(self.iter)
        else:
            self.logdir = self.basedir + '/test/'+str(self.iter)
        if saveimgs and ((Training and self.iter % 20 == 0) or (not Training)) and (not os.path.exists(self.logdir)):
            os.makedirs(self.logdir)


    def close(self):
        self.carla.stop()
        print('stop finish')


    def start(self, rng):

        self.carla = CARLA(self._host, self._port)
        self._reset(rng)
        return self.episode_config[0]

    def reset(self, rng):

        #self.carla = CARLA(self._host, self._port)
        self._reset(rng)
        return self.episode_config[0]
    
    def get_readout_maps(self, pos):
        readout_maps = []
        px = pos.location.x
        py = pos.location.y
        
        ori_x = pos.orientation.x
        ori_y = pos.orientation.y
        ori_vec_length = np.sqrt(ori_x ** 2 + ori_y ** 2)
        ori_x /= ori_vec_length
        ori_y /= ori_vec_length
        id = 0
        for sc in self._map_scales:
            readout_map = []
            hx = self._map_crop_sizes[id]/2
            hy = self._map_crop_sizes[id]/2
            for i in range(self._map_crop_sizes[id]):
                readout_map.append([])
                for j in range(self._map_crop_sizes[id]):
                    dx = (j+0.5-hx)/sc
                    dy = -(i+0.5-hy)/sc
                    world_x = px + (dy*ori_x + dx*(-ori_y))
                    world_y = py + (dy*ori_y + dx*(ori_x))
                    pixel_x = np.floor(((world_x+2137.5) / 16.627)).astype(np.int32)
                    pixel_y = np.floor(((world_y+1675.8) / 16.627)).astype(np.int32)
                    #print([world_x, world_x, pixel_x, pixel_y])
                    if pixel_y >= 0 and pixel_y < len(img_array) and pixel_x >=0 and pixel_x < len(img_array[pixel_y]) and img_array[pixel_y][pixel_x][0] + img_array[pixel_y][pixel_x][1] + img_array[pixel_y][pixel_x][2] > 0: 
                        fs = 0.0
                    else:
                        fs = 1.0
                    readout_map[i].append([fs])
            readout_maps.append([readout_map])
            #print(readout_map)
            id += 1

            '''if(id == 1):
                print([px,py,ori_x,ori_y])
                r_0 = np.concatenate((np.array(readout_map), np.array(readout_map), np.array(readout_map)), 2) * 255.0
                #print(np.uint8(r_0))
                dir2 = self.logdir+"/r_"+str((self.id)) +"_" + str((time.time())) + ".jpg"
                #print(dir2)
                Image.fromarray(np.uint8(r_0)).save(dir2)'''
        return readout_maps
            


    def get_rel_goal_loc(self, current_pos, goal_pos):
        #current_pos = self.positions[self.episode_config[0]]
        #goal_pos = self.positions[self.episode_config[1]]

        #return [[rel_x, rel_y, cos, sin]]

        #print goal_pos.location.x, goal_pos.location.y, goal_pos.orientation.x, goal_pos.orientation.y
        #print current_pos.location.x, current_pos.location.y, current_pos.orientation.x, current_pos.orientation.y

        rel_goal_loc = []

        dy = 0 - (goal_pos.location.y - current_pos.location.y)
        dx = (goal_pos.location.x - current_pos.location.x)
        orientation_vec_square = np.square(current_pos.orientation.x)+np.square(current_pos.orientation.y)
        orientation_vec_length = np.sqrt(orientation_vec_square)
        goal_orientation_vec_square = np.square(goal_pos.orientation.x)+np.square(goal_pos.orientation.y)
        goal_orientation_vec_length = np.sqrt(goal_orientation_vec_square)
        rel_goal_loc.append(((-current_pos.orientation.y)*dx-current_pos.orientation.x*dy)/orientation_vec_length)
        rel_goal_loc.append((current_pos.orientation.x*dx+(-current_pos.orientation.y)*dy)/orientation_vec_length)
        rel_goal_loc.append(current_pos.orientation.x*goal_pos.orientation.x/(orientation_vec_length*goal_orientation_vec_length)+(-current_pos.orientation.y)*(-goal_pos.orientation.y)/(orientation_vec_length*goal_orientation_vec_length))
        rel_goal_loc.append((-goal_pos.orientation.y)*current_pos.orientation.x/(orientation_vec_length*goal_orientation_vec_length)-goal_pos.orientation.x*(-current_pos.orientation.y)/(orientation_vec_length*goal_orientation_vec_length))
        
        #print rel_goal_loc

        return rel_goal_loc

    def get_goal_loc_at_start(self):
        self.start_pos = self.positions[self.episode_config[0]]
        self.goal_pos = self.positions[self.episode_config[1]]
        self.rel_goal_loc_at_start = self.get_rel_goal_loc(self.start_pos, self.goal_pos)
        goal_loc = [self.goal_pos.location.x, self.goal_pos.location.y]
        return self.rel_goal_loc_at_start, goal_loc
    
    def get_ego_goal_img(self):
        current_pos = self._latest_measurements['PlayerMeasurements'].transform
        rel_goal_loc_at_current = self.get_rel_goal_loc(current_pos, self.goal_pos)
        rel_orientation_x = rel_goal_loc_at_current[2]
        rel_orientation_y = rel_goal_loc_at_current[3]
        rel_goal_theta = np.arctan2(rel_orientation_y, rel_orientation_x)
        rel_goal_orientation = 0
        pi = np.pi

        #!!!!!!!!!!TODO:mhr: not sure about the order
        if (rel_goal_theta > np.pi/4):
            if (rel_goal_theta > 3*pi/4):
                rel_goal_orientation = 2
            else:
                rel_goal_orientation = 3
        elif (rel_goal_theta <= -1*pi/4):
            if (rel_goal_theta <= -3*pi/4):
                rel_goal_orientation = 2
            else:
                rel_goal_orientation = 1

        goals = []
        print rel_goal_loc_at_current[0]
        print rel_goal_loc_at_current[1]
        for i, (sc, map_crop_size) in enumerate(zip(self._map_scales, self._map_crop_sizes)):
            x = rel_goal_loc_at_current[0]*sc + (map_crop_size-1.)/2.
            y = rel_goal_loc_at_current[1]*sc + (map_crop_size-1.)/2.
            goal_i = np.zeros((1, map_crop_size, map_crop_size, self._n_ori), dtype=np.float32)
            gc = rel_goal_orientation
            x0 = np.floor(x).astype(np.int32)
            x1 = x0 + 1
            y0 = np.floor(y).astype(np.int32)
            y1 = y0 + 1
            if x0 >= 0 and x0 <= map_crop_size-1:
                if y0 >= 0 and y0 <= map_crop_size-1:
                    goal_i[0, map_crop_size-1-y0, x0, gc] = (x1-x)*(y1-y)
                if y1 >= 0 and y1 <= map_crop_size-1:
                    goal_i[0, map_crop_size-1-y1, x0, gc] = (x1-x)*(y-y0)

            if x1 >= 0 and x1 <= map_crop_size-1:
                if y0 >= 0 and y0 <= map_crop_size-1:
                    goal_i[0, map_crop_size-1-y0, x1, gc] = (x-x0)*(y1-y)
                if y1 >= 0 and y1 <= map_crop_size-1:
                    goal_i[0, map_crop_size-1-y1, x1, gc] = (x-x0)*(y-y0)
            goals.append(goal_i)
        return goals



    def _reset(self, rng):
        print('reset')
        self._start_time = time.time()
        self.positions = self.carla.loadConfigurationFile(self._config_path)
        self.episode_config = find_valid_episode_position(
            self.positions, self._agent.waypointer, rng, self._difficulty)
        self._agent = Agent(ConfigAgent(self._driver_conf.city_name, self._driver_conf.stop4TL))

        self.carla.newEpisode(self.episode_config[0])

        print 'RESET ON POSITION ', self.episode_config[0]
        
        return self.episode_config
    
    def skip_frames(self):
        print ('skip')
        for i in range(self._skiped_frames):
            measurements = self.carla.getMeasurements()
            capture_time = time.time()
            image = measurements['BGRA'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = image[:, :, ::-1]
            image = scipy.misc.imresize(image, [self._driver_conf.resolution[0], self._driver_conf.resolution[1]])
            #Image.fromarray(image).save(self.logdir+"/pre_img_"+str((self.id)) +"_"+str(i) + "_"+str((capture_time)) + ".jpg")
            player_data = measurements['PlayerMeasurements']
            pos = [player_data.transform.location.x, player_data.transform.location.y, 22]
            ori = [player_data.transform.orientation.x, player_data.transform.orientation.y, player_data.transform.orientation.z]
            control = self._agent.get_control(measurements, self.positions[self.episode_config[1]])
            action_noisy, drifting_time, will_drift = self.noiser.compute_noise(control, measurements['PlayerMeasurements'].forward_speed)
            self.act_once([action_noisy.steer, action_noisy.throttle, action_noisy.brake])



    '''def get_recording(self):
        #if self._autopilot:
		if self._skiped_frames >= 20:
			return True
		else:
			self._skiped_frames += 1
			return False'''

    '''def get_reset(self):
        if time.time() - self._start_time > self._reset_period:
            self._reset()
        elif self._latest_measurements['PlayerMeasurements'].collision_vehicles > 0.0 or self._latest_measurements[
            'PlayerMeasurements'].collision_pedestrians > 0.0 or self._latest_measurements[
            'PlayerMeasurements'].collision_other > 0.0:
            self._reset()'''

    def get_waypoints(self):

        wp1, wp2 = self._agent.get_active_wps()
        return [wp1, wp2]

    def get_sensor_data(self):
        measurements = self.carla.getMeasurements()
        self._latest_measurements = measurements
        player_data = measurements['PlayerMeasurements']
        pos = [player_data.transform.location.x,
               player_data.transform.location.y, 22]
        ori = [player_data.transform.orientation.x,
               player_data.transform.orientation.y, player_data.transform.orientation.z]
        
        reach_goal = 0
        if self.reach_goal:
            reach_goal = 1
        elif sldist([player_data.transform.location.x, player_data.transform.location.y],
                    [self.positions[self.episode_config[1]].location.x,
                    self.positions[self.episode_config[1]].location.y]) < self._dist_to_activate:
            reach_goal = 1
            self.reach_goal = True
            #self._reset()  # TODO: delete
		# print 'Selected Position ',self.episode_config[1],'from len ', len(self.positions)

        if reach_goal == 0:
            #directions ( 3 is left, 4 is right, 5 is straight)
            direction, _ = self.planner.get_next_command(pos, ori, [
                self.positions[self.episode_config[1]].location.x, self.positions[self.episode_config[1]].location.y,
                22], (1, 0, 0))
            control = self._agent.get_control(self._latest_measurements, self.positions[self.episode_config[1]])
            action_noisy, drifting_time, will_drift = self.noiser.compute_noise(control, self._latest_measurements['PlayerMeasurements'].forward_speed)
            self.expert_control = action_noisy
        else:
            direction = 3.0
            control = Control()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            action_noisy = control
            self.expert_control = action_noisy
        return measurements, direction, reach_goal, action_noisy, control
        '''or \
                    compute_angle(player_data.transform.orientation.x, player_data.transform.orientation.y,\
                    self.positions[self.episode_config[1]].location.x- player_data.transform.location.x, \
                    self.positions[self.episode_config[1]].location.y- player_data.transform.location.y) < -0.7'''

    def get_data_for_mapper(self, step_number): #, typ='rgb'):

		#recorder = Recorder(drive_config.path + folder_name + '/', drive_config.resolution, \
        #                    image_cut=drive_config.image_cut, camera_dict=camera_dict, record_waypoints=True)
        measurements, direction, reach_goal, action_noisy, control = self.get_sensor_data()
        capture_time = time.time()
        if self.typ == 'rgb':
            image = measurements['BGRA'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = image[:, :, ::-1]
            image = scipy.misc.imresize(image, [self._driver_conf.resolution[0], self._driver_conf.resolution[1]])
            #if step_number % 4 == 0 or step_number==79:
            if saveimgs and ((not Training) or (Training and self.iter % 20 == 0)):
                Image.fromarray(image).save(self.logdir+"/img_"+str((self.id)) +"_" + str((capture_time)) + ".jpg")
            #image_input = image *1. - 128
        elif self.typ == 'd':
            image = measurements['Depth'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = scipy.misc.imresize(image, [self._driver_conf.resolution[0], self._driver_conf.resolution[1]])
            if saveimgs and (step_number % 4 == 0 or step_number==79):
                Image.fromarray(image).save(self.logdir+"/dep_" +str((self.id)) +"_" + str((capture_time)) + ".jpg")
            #image_input = np.array(image)
        else:
            logging.fatal('Sampling not one of uniform.')
        return image, measurements, direction, reach_goal, action_noisy, control

    def act_once(self, action):
        control = Control()
        print(action)
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
        self.carla.sendCommand(control)

    def act(self, action):
        control = Control()
        print(action)
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
        #print(control.throttle)
        for i in range(self._replay_action - 1):
            self.carla.sendCommand(control)
            self.carla.getMeasurements()
        self.carla.sendCommand(control)
	
        '''self.rel_goal_loc_at_start = []
        dy = self.goal_pos.location.y - self.start_pos.location.y
        dx = self.goal_pos.location.x - self.start_pos.location.x
        orientation_vec_square = np.square(self.start_pos.orientation.x)+np.square(self.start_pos.orientation.y)
        orientation_vec_length = np.sqrt(orientation_vec_square)
        goal_orientation_vec_square = np.square(self.goal_pos.orientation.x)+np.square(self.goal_pos.orientation.y)
        goal_orientation_vec_length = np.sqrt(goal_orientation_vec_square)
        self.rel_goal_loc_at_start.append((self.start_pos.orientation.y*dx-self.start_pos.orientation.x*dy)/orientation_vec_length)
        self.rel_goal_loc_at_start.append((self.start_pos.orientation.x*dx+self.start_pos.orientation.y*dy)/orientation_vec_length)
        self.rel_goal_loc_at_start.append(self.start_pos.orientation.x*self.goal_pos.orientation.x/(orientation_vec_length*goal_orientation_vec_length)+self.start_pos.orientation.y*self.goal_pos.orientation.y/(orientation_vec_length*goal_orientation_vec_length))
        self.rel_goal_loc_at_start.append(self.goal_pos.orientation.y*self.start_pos.orientation.x/(orientation_vec_length*goal_orientation_vec_length)-self.goal_pos.orientation.x*self.start_pos.orientation.y/(orientation_vec_length*goal_orientation_vec_length))
        '''


'''
class Manual(AgentBench):
    """
    Sample redefinition of the Agent,
    An agent that goes straight
    """
    def run_step(self, measurements, sensor_data, target):
        control = VehicleControl()
        control.throttle = 0.9

        return control

class Machine(AgentBench):
  def initialize(self):
      pass

  def run_step(self, measurements, sensor_data, target):
      control = VehicleControl()
      control.throttle = 0.9

      return control



city_name = 'Town02'
host = 'localhost'
port = 2012
log_name = 'test'
while True:
    try:
        with make_carla_client(host, port) as client:
            corl = CoRL2017(city_name=city_name, name_to_save=log_name)
            agent = Machine(city_name)
            agent.initialize()
            results = corl.benchmark_agent(agent, client)
            corl.plot_summary_test()
            corl.plot_summary_train()

            break

    except TCPConnectionError as error:
        logging.error(error)
        time.sleep(1)'''