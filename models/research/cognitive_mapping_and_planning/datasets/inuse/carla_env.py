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

#TODO: action_noise 并未用rng


from datasets.inuse.drive_interfaces.driver import Driver
from datasets.inuse.drive_interfaces.configuration.cognitive_drive_config import configDrive
from datasets.inuse.drive_interfaces.noiser import Noiser
# from datasets.inuse.drive_interfaces.carla.comercial_cars.carla_human import CarlaHuman
from datasets.inuse.drive_interfaces.carla.carla_client.carla.carla import CARLA, Measurements, Control
from datasets.inuse.drive_interfaces.carla.carla_client.carla.agent import *
from datasets.inuse.drive_interfaces.carla.carla_client.carla import Planner




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


def find_valid_episode_position(positions, waypointer, rng):
    found_match = False
    while not found_match:
        index_start = rng.choice(range(len(positions))) #np.random.randint(len(positions))
        start_pos = positions[index_start]
        if not waypointer.test_position((start_pos.location.x, start_pos.location.y, 22),
                                        (start_pos.orientation.x, start_pos.orientation.y, start_pos.orientation.z)):
            continue
        index_goal = rng.choice(range(len(positions))) #np.random.randint(len(positions))
        if index_goal == index_start:
            continue

        print (' TESTING (', index_start, ',', index_goal, ')')
        goals_pos = positions[index_goal]
        if not waypointer.test_position((goals_pos.location.x, goals_pos.location.y, 22),
                                        (goals_pos.orientation.x, goals_pos.orientation.y, goals_pos.orientation.z)):
            continue
        if sldist([start_pos.location.x, start_pos.location.y], [goals_pos.location.x, goals_pos.location.y]) < 25000.0:
            print ('COntinued on distance ', sldist([start_pos.location.x, start_pos.location.y], [
                goals_pos.location.x, goals_pos.location.y]))

            continue

        if waypointer.test_pair((start_pos.location.x, start_pos.location.y, 22),
                                (start_pos.orientation.x, start_pos.orientation.y, start_pos.orientation.z),
                                (goals_pos.location.x, goals_pos.location.y, 22)):
            found_match = True
        waypointer.reset()
    waypointer.reset()

    return index_start, index_goal


class CarlaEnvMultiplexer:
    def __init__(self):
        self.drive_config = configDrive()
        # TODO: adding more carla_configs
        prefix = './datasets/inuse/drive_interfaces/carla/'
        self.carla_configs = [prefix + 'CarlaSettings3CamTest.ini']
        #self.driver = self.get_instance()

    def sample_env(self, rngs):
        rng = rngs[0]
        self.drive_configs = [self.drive_config, self.drive_config, self.drive_config, self.drive_config]
        for i in range(4):
            self.drive_configs[i].carla_config = rng.choice(self.carla_configs)
        self.carla_env_wrapper = CarlaEnvWrapper(self.drive_configs)
        return self.carla_env_wrapper



class CarlaEnvWrapper():

    #!!!!!mhr: TODO: reset the environment

    def __init__(self, drive_configs):
        self.drive_configs = drive_configs
        self.carla_envs = []
        for i in range(len(drive_configs)):
            self.carla_envs.append(self.create_env(drive_configs[i]))
        self.history = [[],[],[],[]]

    def reset(self, rngs):
        start_indices = []
        rng = rngs[0]
        for i in range(len(self.carla_envs)):
            start_index = self.carla_envs[i].start(rng)
            start_indices.append(start_index)
        return start_indices

    def create_env(self, drive_config):
        
        driver = CarlaEnv(drive_config)
        self.camera_dict = get_camera_dict(drive_config.carla_config)
        print " Camera Dict "
        print self.camera_dict
        # folder_name = str(datetime.datetime.today().year) + str(datetime.datetime.today().month) + str(datetime.datetime.today().day)
        return driver

    def get_common_data(self):
        maps = [[],[],[],[]]
        line = []
        for i in range(524):
            line.append([0.0])
        map = []
        for i in range(1112):
            map.append(line)
        for i in range(4):
            maps[i].append(map)

        rel_goal_locs = []
        goal_locs = []
        goal_locs.append([])
        for i in range(len(self.carla_envs)):
            rel_goal_loc, goal_loc = self.carla_envs[i].get_goal_loc_at_start()
            rel_goal_locs.append([rel_goal_loc])
            goal_locs[0].append([goal_loc])
        maps = np.array(maps).astype(np.float32)
        goal_locs = (np.array(goal_locs)).astype(np.float32)
        rel_goal_locs = (np.array(rel_goal_locs)).astype(np.float32)
        vars(utils.Foo(orig_maps=maps, goal_loc=goal_locs, rel_goal_loc_at_start=rel_goal_locs))

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
        outs['ego_goal_imgs_0'] = []
        outs['ego_goal_imgs_1'] = []
        outs['ego_goal_imgs_2'] = []
        outs['incremental_thetas'] = []
        outs['imgs'] = []

        #useless inputs
        outs['node_ids'] = []
        outs['gt_dist_to_goal'] = []
        
        
        for i in range(len(self.carla_envs)):
            image, measurements, direction, reach_goal, action_noisy, control = self.carla_envs[i].get_data_for_mapper()
            self.history[i].append([image, measurements, direction, reach_goal, action_noisy, control])
            outs['loc_on_map'].append([[measurements['PlayerMeasurements'].transform.location.x, measurements['PlayerMeasurements'].transform.location.y]])
            goal_imgs = self.carla_envs[i].get_ego_goal_img()
            for i in len(goal_imgs):
                outs['ego_goal_imgs_{:d}'.format(i)].append(goal_imgs[i])
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
            outs['incremental_thetas'].append([translation_theta - previous_theta, current_theta - translation_theta])
            outs['imgs'].append([[self.image_preprocess(image)]])
            square = np.square(measurements['PlayerMeasurements'].transform.location.y-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.y) +np.square(measurements['PlayerMeasurements'].transform.location.x-self.history[i][step_number-1][1]['PlayerMeasurements'].transform.location.x)
            length = np.sqrt(square)
            outs['incremental_locs'].append([[0.0, length]])

            #useless inputs
            outs['node_ids'].append([[self.carla_envs[i].episode_config[0]]]) 
            outs['gt_dist_to_goal'].append([[0.0]]) 
            outs['perturbs'].append([[0.0, 0.0, 0.0, 0.0]]) 

        for key in outs.keys():
            outs[key] = np.array(outs[key]).astype(np.float32)
        return outs

    def get_optimal_action(self, current_node_ids, step_number):
        """Returns the optimal action from the current node."""
        optimal_actions = []
        for i in range(len(self.carla_envs)):
            optimal_actions.append([self.history[i][step_number][4].steer, self.history[i][step_number][4].throttle, self.history[i][step_number][4].brake])
        optimal_actions = np.array(optimal_actions).astype(np.float32)
        return optimal_actions

    def get_targets(self, current_node_ids, step_number):
        """Returns the target actions from the current node."""
        action = self.get_optimal_action(current_node_ids, step_number)
        action = np.expand_dims(action, axis=1)
        return vars(utils.Foo(action=action))

    def take_action(self, current_node_ids, action, step_number):
        """In addition to returning the action, also returns the reward that the
        agent receives."""
        #mhr: TODO:improve performance by reducing the times to get sensor data
        
        #return starting node id
        starting_indices = [self.carla_envs]
        rewards = []

        for i in range(len(self.carla_envs)):
            self.carla_envs[i].act(action[i])
            measurements, direction, reach_goal, action_noisy, control = self.carla_envs[i].get_sensor_data()
            reward = 0
            if reach_goal:
                reward = self.drive_configs[i].reward_at_goal
            reward -= self.drive_configs[i].reward_time_penalty
            rewards.append(reward)
            starting_indices.append(self.carla_envs[i].episode_config[0])
        return starting_indices, rewards
        
    def get_features_name(self):
        f = []
        f.append('imgs')
        f.append('rel_goal_loc')
        f.append('loc_on_map')
        f.append('gt_dist_to_goal')
        for i in range(len(self.task_params.map_scales)):
            f.append('ego_goal_imgs_{:d}'.format(i))
        f.append('incremental_locs')
        f.append('incremental_thetas')
        f.append('node_ids')
        f.append('perturbs')
        return f

class CarlaEnv(Driver):

    def __init__(self, driver_conf):
        Driver.__init__(self)
        # self._straight_button = False
        # self._left_button = False
        # self._right_button = False
        # self._recording= False

        self._skiped_frames = 20  # ??? TODO: delete it

        # load a manager to deal with test data
        # self.use_planner = driver_conf.use_planner

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
        self._agent = Agent(ConfigAgent(driver_conf.city_name))
        self.noiser = Noiser(driver_conf.noise)
        self._map_scales = driver_conf.map_scales
        self._map_crop_sizes = driver_conf.map_crop_sizes
        self._n_ori = driver_conf.n_ori

    #def 


    def start(self, rng):

        self.carla = CARLA(self._host, self._port)
        self._reset(rng)
        return self.episode_config[0]




    def get_rel_goal_loc(self, current_pos, goal_pos):
        #current_pos = self.positions[self.episode_config[0]]
        #goal_pos = self.positions[self.episode_config[1]]

        #return [[rel_x, rel_y, cos, sin]]

        rel_goal_loc = []

        dy = goal_pos.location.y - current_pos.location.y
        dx = goal_pos.location.x - current_pos.location.x
        orientation_vec_square = np.square(current_pos.orientation.x)+np.square(current_pos.orientation.y)
        orientation_vec_length = np.sqrt(orientation_vec_square)
        goal_orientation_vec_square = np.square(goal_pos.orientation.x)+np.square(goal_pos.orientation.y)
        goal_orientation_vec_length = np.sqrt(goal_orientation_vec_square)
        rel_goal_loc.append((current_pos.orientation.y*dx-current_pos.orientation.x*dy)/orientation_vec_length)
        rel_goal_loc.append((current_pos.orientation.x*dx+current_pos.orientation.y*dy)/orientation_vec_length)
        rel_goal_loc.append(current_pos.orientation.x*goal_pos.orientation.x/(orientation_vec_length*goal_orientation_vec_length)+current_pos.orientation.y*goal_pos.orientation.y/(orientation_vec_length*goal_orientation_vec_length))
        rel_goal_loc.append(goal_pos.orientation.y*current_pos.orientation.x/(orientation_vec_length*goal_orientation_vec_length)-goal_pos.orientation.x*current_pos.orientation.y/(orientation_vec_length*goal_orientation_vec_length))
        return self.rel_goal_loc_at_start

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
        for i, (sc, map_crop_size) in enumerate(zip(self._map_scales, self._map_crop_sizes)):
            x = rel_goal_loc_at_current[0]*sc + (map_crop_size-1.)/2.
            y = rel_goal_loc_at_current[0]*sc + (map_crop_size-1.)/2.
            goal_i = np.zeros((1, map_crop_size, map_crop_size, self._n_ori), dtype=np.float32)
            gc = rel_goal_orientation
            x0 = np.floor(x).astype(np.int32)
            x1 = x0 + 1
            y0 = np.floor(y).astype(np.int32)
            y1 = y0 + 1
            if x0 >= 0 and x0 <= map_crop_size-1:
                if y0 >= 0 and y0 <= map_crop_size-1:
                    goal_i[0, y0, x0, gc] = (x1-x)*(y1-y)
                if y1 >= 0 and y1 <= map_crop_size-1:
                    goal_i[0, y1, x0, gc] = (x1-x)*(y-y0)

            if x1 >= 0 and x1 <= map_crop_size-1:
                if y0 >= 0 and y0 <= map_crop_size-1:
                    goal_i[0, y0, x1, gc] = (x-x0)*(y1-y)
                if y1 >= 0 and y1 <= map_crop_size-1:
                    goal_i[0, y1, x1, gc] = (x-x0)*(y-y0)
            goals.append(goal_i)
        return goals



    def _reset(self, rng):
        self._start_time = time.time()
        self.positions = self.carla.loadConfigurationFile(self._config_path)
        self.episode_config = find_valid_episode_position(
            self.positions, self._agent.waypointer, rng)
        self._agent = Agent(ConfigAgent(self._driver_conf.city_name))

        self.carla.newEpisode(self.episode_config[0])

        print 'RESET ON POSITION ', self.episode_config[0]
        self._dist_to_activate = 300  # TODO:?
        # Set 10 frames to skip
        self._skiped_frames = 10  # TODO:?
        return self.episode_config

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
        if sldist([player_data.transform.location.x, player_data.transform.location.y],
					[self.positions[self.episode_config[1]].location.x,
					self.positions[self.episode_config[1]].location.y]) < self._dist_to_activate:
			reach_goal = 1
			#self._reset()  # TODO: delete
		# print 'Selected Position ',self.episode_config[1],'from len ', len(self.positions)

        direction, _ = self.planner.get_next_command(pos, ori, [
			self.positions[self.episode_config[1]].location.x, self.positions[self.episode_config[1]].location.y,
			22], (1, 0, 0))
       	control = self._agent.get_control(
        self._latest_measurements, self.positions[self.episode_config[1]])
        action_noisy, drifting_time, will_drift = self.noiser.compute_noise(control, self._latest_measurements['PlayerMeasurements'].forward_speed)
        self.expert_control = action_noisy
        return measurements, direction, reach_goal, action_noisy, control

    def get_data_for_mapper(self): #, typ='rgb'):

		#recorder = Recorder(drive_config.path + folder_name + '/', drive_config.resolution, \
        #                    image_cut=drive_config.image_cut, camera_dict=camera_dict, record_waypoints=True)
        measurements, direction, reach_goal, action_noisy, control = self.get_sensor_data()
        capture_time = time.time()
        if self.typ == 'rgb':
            image = measurements['BGRA'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = image[:, :, ::-1]
            image = scipy.misc.imresize(image, [self._driver_conf._resolution[0], self._driver_conf._resolution[1]])
            Image.fromarray(image).save("datasets/inuse/cog/img_" + str((capture_time)) + ".png")
            #image_input = image *1. - 128
        elif self.typ == 'd':
            image = measurements['Depth'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = scipy.misc.imresize(image, [self._driver_conf._resolution[0], self._driver_conf._resolution[1]])
            Image.fromarray(image).save("datasets/inuse/cog/dep_" + str((capture_time)) + ".png")
            #image_input = np.array(image)
        else:
            logging.fatal('Sampling not one of uniform.')
        return image, measurements, direction, reach_goal, action_noisy, control


    def act(self, action):
        control = Control()
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
        self.carla.sendCommand(action)
	
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

