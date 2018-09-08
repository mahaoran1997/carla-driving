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

#TODO: action_noise does not use rng


from datasets.inuse.drive_interfaces.driver import Driver
from datasets.inuse.drive_interfaces.configuration.cognitive_drive_config import configDrive
from datasets.inuse.drive_interfaces.noiser import Noiser
# from datasets.inuse.drive_interfaces.carla.comercial_cars.carla_human import CarlaHuman
from datasets.inuse.drive_interfaces.carla.carla_client.carla.carla import CARLA, Measurements, Control
from datasets.inuse.drive_interfaces.carla.carla_client.carla.agent import *
from datasets.inuse.drive_interfaces.carla.carla_client.carla import Planner


#tsu  scylla

#asp:128.32.255.29  namazu:128.32.255.26 basilisk:128.32.111.70

Training =False
batch_size = 1
cc_num = 15
hosts = ['128.32.255.26']
port=2050
if Training:
    batch_size = 18
    port=2000
    hosts = ['128.32.255.29', '128.32.255.26', '128.32.111.70']
    cityfile = 'datasets/inuse/drive_interfaces/carla/carla_client/carla/planner/carla_1.png'
else:
    cityfile= 'datasets/inuse/drive_interfaces/carla/carla_client/carla/planner/carla_1.png'
readout_img = Image.open(cityfile)
img_array = np.asarray(readout_img)

readout = False

saveimgs = True






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

def find_valid_episode_position(positions, waypointer, rng):

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
    def __init__(self, logdir, city_id):
        #self.drive_config = configDrive()
        self.drive_configs = []
        for i in range(batch_size):
            self.drive_configs.append(configDrive())
        self.carla_env_wrapper = None
        self.logdir = logdir
        self.city_id = city_id

    def sample_env(self, rngs):
        #get carla_env_wrapper
        rng = rngs[0]
        id = 0
        for host_i in hosts:
            for i in range(batch_size/len(hosts)):
                self.drive_configs[id].host = host_i
                self.drive_configs[id].port = port + i * 3    
                id += 1
        if self.carla_env_wrapper == None:
            self.carla_env_wrapper = CarlaEnvWrapper(self.drive_configs, self.logdir, self.city_id, rng)
        return self.carla_env_wrapper
    



class CarlaEnvWrapper():

    def __init__(self, drive_configs, logdir, city_id, rng):
        self.drive_configs = drive_configs
        self.carla_envs = []
        self.logdir = logdir
        for i in range(len(drive_configs)):
            print ('Environment {:d}'.format(i))
            self.carla_envs.append(CarlaEnv(drive_configs[i], i, logdir, city_id, rng))
    

    def get_features_name(self):
        f = []
        f.append('imgs')
        if readout:
            for i in range(len(self.drive_configs[0].map_scales)):
                f.append('readout_maps_{:d}'.format(i))
            for i in range(len(self.drive_configs[0].map_scales)):
                f.append('ego_goal_imgs_{:d}'.format(i))
        f.append('speed')
        f.append('command')
        f.append('incremental_locs')
        f.append('incremental_thetas')
        f.append('running_sum_num')
        return f

    def get_features(self):

        outs = {}
        for env in self.carla_envs:
            env_outs = env.get_data()
            #print(env_outs)
            for (k, v) in env_outs.items():
                if not (k in outs):
                    outs[k] = []
                outs[k].append(v)
        for key in outs.keys():
            outs[key] = np.array(outs[key]).astype(np.float32)
        return outs

    
    def get_optimal_action(self):
        """Returns the optimal action from the current node."""
        optimal_actions = []
        for i in range(len(self.carla_envs)):
            optimal_control = self.carla_envs[i].get_control()
            optimal_actions.append([optimal_control.steer, optimal_control.throttle, optimal_control.brake])
        optimal_actions = np.array(optimal_actions).astype(np.float32)
        return optimal_actions

    def get_targets(self):
        optimal_actions = self.get_optimal_action()
        
        return vars(utils.Foo(action=optimal_actions))
    
    def get_targets_name(self):
        return ['action']

    def take_action(self, action, running_sum_nums):
        #print(running_sum_nums)
        for i in range(len(self.carla_envs)):
            self.carla_envs[i].act(action[i], running_sum_nums[i])
        resets = []
        for i in range(len(self.carla_envs)):
            resets.append(self.carla_envs[i].update())
        return resets
        
    


class CarlaEnv(Driver):

    def __init__(self, driver_conf, env_id, logdir, city_id, rng):
        Driver.__init__(self)
        self.iter = -1
        self.logdir = logdir
        self.basedir = logdir
        self.id = env_id
        self.rng = rng
        
        prefix = './datasets/inuse/drive_interfaces/carla/CarlaSettingCity'+str(city_id)+'W'
        self.carla_configs = []
        for i in range(cc_num):
            self.carla_configs.append(prefix+str(i)+'.ini')
        self.planner = Planner('./datasets/inuse/drive_interfaces/carla/comercial_cars/' + driver_conf.city_name +
                               '.txt', './datasets/inuse/drive_interfaces/carla/comercial_cars/' + driver_conf.city_name + '.png')
        self._host = driver_conf.host
        self._port = driver_conf.port
        self._resolution = driver_conf.resolution
        self._image_cut = driver_conf.image_cut
        self._reset_period = driver_conf.reset_period
        self._driver_conf = driver_conf
        self.typ = driver_conf.typ
        self._rear = False
        self._agent = Agent(ConfigAgent(driver_conf.city_name, driver_conf.stop4TL))
        self.noiser = Noiser(driver_conf.noise)
        self._map_scales = driver_conf.map_scales
        self._map_crop_sizes = driver_conf.map_crop_sizes
        self._n_ori = driver_conf.n_ori
        self._stop4TL = driver_conf.stop4TL
        #self._resolution = driver_conf.resolution
        self._dist_to_activate = driver_conf.dist_to_activate # TODO:?
        # Set 10 frames to skip
        self._skiped_frames = driver_conf.skiped_frames  # TODO:?
        self._replay_action = driver_conf.replay_action
        self.reach_goal_times = 0
        self.collision_times = 0
        #self.reach_goal = False
        self.carla = None
        self.saveimgs = False
        self.need_reset = True
        
        
    def _reset(self):
        print('reset')
        self.saveimgs = False
        if self.carla == None:
            self.carla = CARLA(self._host, self._port)

        self.iter += 1
        if Training:
            self.logdir = self.basedir + '/'+ str(self.iter) + '/' + str(self.id)
        else:
            self.logdir = self.basedir + '/' + str(self.iter) + '/' + str(self.id)
        if saveimgs and ((Training and self.iter % 10 == 0) or (not Training)):
            if (not os.path.exists(self.logdir)):
                os.makedirs(self.logdir)
            self.saveimgs = True

        self.prev_data = None
        self.current_data = None
        
        

        self.running_sum_num = np.zeros((64, 64, 32))
        
        self._start_time = time.time()
        self._config_path = self.rng.choice(self.carla_configs)
        self.positions = self.carla.loadConfigurationFile(self._config_path)
        self.episode_config = find_valid_episode_position(
            self.positions, self._agent.waypointer, self.rng)
        self._agent = Agent(ConfigAgent(self._driver_conf.city_name, self._driver_conf.stop4TL))
        self.carla.newEpisode(self.episode_config[0])
        print 'RESET ON POSITION ', self.episode_config[0]
        self.step_number = 0
        self.skip_frames()
        self.need_reset = False


        self.step_number = 0
        
    def skip_frames(self):
        print ('skip')
        for i in range(self._skiped_frames):
            measurements = self.carla.getMeasurements()
            control = self._agent.get_control(measurements, self.positions[self.episode_config[1]])
            action_noisy, drifting_time, will_drift = self.noiser.compute_noise(control, measurements['PlayerMeasurements'].forward_speed)
            self.skip_act([action_noisy.steer, action_noisy.throttle, action_noisy.brake])
        self.current_data = self.get_sensor_data()

    def get_control(self):
        return self.current_data[2]

    def get_data(self):
        if self.need_reset:
            self._reset()
        measurements = self.current_data[0]
        capture_time = time.time()
        if self.typ == 'rgb':
            image = measurements['BGRA'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = image[:, :, ::-1]
            image = scipy.misc.imresize(image, [self._driver_conf.resolution[0], self._driver_conf.resolution[1]])
            #if step_number % 4 == 0 or step_number==79:
            if self.saveimgs:
                Image.fromarray(image).save(self.logdir+"/img_"+str((self.id)) +"_" + str((capture_time)) + ".jpg")
            #image_input = image *1. - 128
        elif self.typ == 'd':
            image = measurements['Depth'][0][self._driver_conf.image_cut[0]:self._driver_conf.image_cut[1], self._driver_conf.image_cut[2]:self._driver_conf.image_cut[3], :3]
            image = scipy.misc.imresize(image, [self._driver_conf.resolution[0], self._driver_conf.resolution[1]])
            if self.saveimgs:
                Image.fromarray(image).save(self.logdir+"/dep_" +str((self.id)) +"_" + str((capture_time)) + ".jpg")
            #image_input = np.array(image)
        else:
            logging.fatal('Sampling not one of uniform.')
        
        outs = {}
        outs['imgs'] = [self.image_preprocess(image)]
        outs['running_sum_num'] = self.running_sum_num
        outs['command'] = [0.0, 0.0, 0.0, 0.0]
        outs['command'][int(round(self.current_data[1]-2.0))] = 1.0
        outs['speed'] = [measurements['PlayerMeasurements'].forward_speed]

        current_theta = np.arctan2(measurements['PlayerMeasurements'].transform.orientation.y, measurements['PlayerMeasurements'].transform.orientation.x)
        
        if (self.step_number > 0):
            history_measurements = self.prev_data[0]
            square = np.square(measurements['PlayerMeasurements'].transform.location.y-history_measurements['PlayerMeasurements'].transform.location.y) +np.square(measurements['PlayerMeasurements'].transform.location.x-history_measurements['PlayerMeasurements'].transform.location.x)
            length = np.sqrt(square)
            if measurements['PlayerMeasurements'].transform.location.y-history_measurements['PlayerMeasurements'].transform.location.y < 0.1 and measurements['PlayerMeasurements'].transform.location.x-history_measurements['PlayerMeasurements'].transform.location.x < 0.1:
                translation_theta = current_theta
            else:
                translation_theta = np.arctan2(measurements['PlayerMeasurements'].transform.location.y-history_measurements['PlayerMeasurements'].transform.location.y, measurements['PlayerMeasurements'].transform.location.x-history_measurements['PlayerMeasurements'].transform.location.x)
            previous_theta = np.arctan2(history_measurements['PlayerMeasurements'].transform.orientation.y, history_measurements['PlayerMeasurements'].transform.orientation.x)
        else:
            translation_theta = current_theta
            previous_theta = current_theta
            length = 0.0
        outs['incremental_thetas'] = [-(translation_theta - previous_theta), -(current_theta - translation_theta)]
        outs['incremental_locs'] = [0.0, length]

        return outs

    def get_sensor_data(self):
        measurements = self.carla.getMeasurements()
        player_data = measurements['PlayerMeasurements']
        pos = [player_data.transform.location.x,
               player_data.transform.location.y, 22]
        ori = [player_data.transform.orientation.x,
               player_data.transform.orientation.y, player_data.transform.orientation.z]
        if sldist([player_data.transform.location.x, player_data.transform.location.y],
                    [self.positions[self.episode_config[1]].location.x,
                    self.positions[self.episode_config[1]].location.y]) < self._dist_to_activate:
            self.reach_goal_times += 1
            self.need_reset = True
        if self.step_number > self._reset_period:
            self.need_reset = True
        elif measurements['PlayerMeasurements'].collision_vehicles > 0.0 or measurements[
            'PlayerMeasurements'].collision_pedestrians > 0.0 or measurements[
            'PlayerMeasurements'].collision_other > 0.0:
            self.need_reset = True
            self.collision_times += 1
        #directions ( 3 is left, 4 is right, 5 is straight)
        direction, _ = self.planner.get_next_command(pos, ori, [
            self.positions[self.episode_config[1]].location.x, self.positions[self.episode_config[1]].location.y,
            22], (1, 0, 0))
        control = self._agent.get_control(measurements, self.positions[self.episode_config[1]])
            #action_noisy, drifting_time, will_drift = self.noiser.compute_noise(control, self._latest_measurements['PlayerMeasurements'].forward_speed)
            #self.expert_control = action_noisy
        return (measurements, direction, control)
        

    def skip_act(self, action,):
        control = Control()
        print(action)
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
        self.carla.sendCommand(control)
    
    def act(self, action, running_sum_num):
        control = Control()
        print(action)
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
        self.carla.sendCommand(control)
        self.step_number += 1
        self.running_sum_num = running_sum_num
        self.prev_data = self.current_data

    def update(self):
        self.current_data = self.get_sensor_data()
        return self.need_reset


    def image_preprocess(self, image):
        if self._driver_conf.typ == 'rgb':
            image = image.astype(np.float32)*1.0
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
        return readout_maps


'''
    
    
        def close(self):
        self.carla.stop()
        print('stop finish')


'''