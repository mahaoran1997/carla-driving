import sys
import pygame
import socket
import random

class ImageData():

    def __init__(self):
        self.raw_rgb = []
        self.rgb = []
        self.depth = []
        self.scene_seg = []


from carla import sensor
from carla.client import CarlaClient, make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from carla.client import VehicleControl as Control

from carla.planner.planner import Planner

from carla.autopilot.autopilot import Autopilot

from carla.autopilot.pilotconfiguration import ConfigAutopilot


from carla.autopilot.waypointer import Waypointer

import numpy as np

sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
import time
from ConfigParser import ConfigParser
from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from contextlib import contextmanager
import sys, os
import copy
import math


def find_valid_episode_position(positions, waypointer):
    found_match = False
    print ('enter check position function')
    while not found_match:
        index_start = np.random.randint(len(positions))
        start_pos = positions[index_start]
        #print(start_pos)
        #print('enter waypointer')
        if not waypointer.test_position((start_pos.location.x, start_pos.location.y, 22)):
            continue
        print('check yes')
        index_goal = np.random.randint(len(positions))
        if index_goal == index_start:
            continue

        print (' TESTING (', index_start, ',', index_goal, ')')
        goals_pos = positions[index_goal]
        if not waypointer.test_position((goals_pos.location.x, goals_pos.location.y, 22)):
            continue
        if sldist([start_pos.location.x, start_pos.location.y],
                  [goals_pos.location.x, goals_pos.location.y]) < 15000.0:  # This distance is arbitrary
            print ('COntinued on distance ',
                   sldist([start_pos.location.x, start_pos.location.y], [goals_pos.location.x, goals_pos.location.y]))

            continue

        if waypointer.test_pair((start_pos.location.x, start_pos.location.y, 22)
                                ,(start_pos.orientation.x, start_pos.orientation.y, start_pos.orientation.z)
                                ,(goals_pos.location.x, goals_pos.location.y, 22)
                                ,(goals_pos.orientation.x, goals_pos.orientation.y, 22)
                                ):
            found_match = True
        waypointer.reset()
    waypointer.reset()

    return index_start, index_goal


class CarlaHuman(object):

    def __init__(self, driver_conf):

        self._straight_button = False
        self._left_button = False
        self._right_button = False
        self._recording = False
        self._skiped_frames = 20

        # load a manager to deal with test data
        self.use_planner = driver_conf.use_planner

        print 'CITY NAME'
        print driver_conf.city_name

        if driver_conf.use_planner:
            self.planner = Planner(driver_conf.city_name)

        self._host = driver_conf.host
        self._port = driver_conf.port
        self._config_path = driver_conf.carla_config
        self._resolution = driver_conf.resolution
        self._autopilot = driver_conf.autopilot
        self._reset_period = driver_conf.reset_period
        self._driver_conf = driver_conf

        self._rear = False

        print ("driver_conf.city_name:")
        print (driver_conf.city_name)

        if self._autopilot:
            self._agent = Autopilot(ConfigAutopilot(driver_conf.city_name))

    def start(self):

        self.carla = CarlaClient(self._host, self._port)
        print ('Try to connect to')
        print (self._host)
        print (self._port)
        self.carla.connect()
        print ('Connected!')
        #while(True):
        #    pass
        #self.carla = make_carla_client(self._host, self._port)
        self._reset()

        if not self._autopilot:
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                print "Please Connect Just One Joystick"
                raise

            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def _reset(self):

        config = ConfigParser()

        config.optionxform = str
        config.read(self._config_path)
        config.set('CARLA/LevelSettings', 'NumberOfVehicles', self._driver_conf.cars)

        config.set('CARLA/LevelSettings', 'NumberOfPedestrians', self._driver_conf.pedestrians)

        config.set('CARLA/LevelSettings', 'WeatherId', self._driver_conf.weather)

        # Write down a temporary init_file to be used on the experiments
        temp_f_name = 'p' + str(self._driver_conf.pedestrians) + '_c' + str(self._driver_conf.cars) + "_w" \
                      + str(self._driver_conf.weather) + '.ini'

        print (temp_f_name)
        print (config)

        with open(temp_f_name, 'w') as configfile:
            config.write(configfile)

        self._start_time = time.time()

        #print (self._config_path)

        with open(self._config_path) as file:
            p = self.carla.load_settings(file.read())
            #print (p)
            self.positions = p.player_start_spots
            number_of_player_starts = len(p.player_start_spots)
        #print (self.positions[0])
        self.episode_config = [random.randint(0, max(0, number_of_player_starts - 1)), random.randint(0, max(0, number_of_player_starts - 1))]
        #print ('after episode_config')
        #self._agent = Autopilot(ConfigAutopilot(self._driver_conf.city_name))
        #print ('before starting episode')
        self.carla.start_episode(self.episode_config[0])
        #print ('after starting episode')
        print 'RESET ON POSITION ', self.episode_config[0]
        measurements, sensor_data = self.carla.read_data()
        print (123)
        print (sensor_data)
        self._dist_to_activate = 300
        # Set 10 frames to skip
        self._skiped_frames = 0

    def _get_direction_buttons(self):
        # with suppress_stdout():
        if (self.joystick.get_button(6)):
            self._left_button = False
            self._right_button = False
            self._straight_button = False

        if (self.joystick.get_button(5)):
            self._left_button = True
            self._right_button = False
            self._straight_button = False

        if (self.joystick.get_button(4)):
            self._right_button = True
            self._left_button = False
            self._straight_button = False

        if (self.joystick.get_button(7)):
            self._straight_button = True
            self._left_button = False
            self._right_button = False

        return [self._left_button, self._right_button, self._straight_button]

    def compute_direction_joystick(self, pos, ori, goal_pos, goal_ori):

        # BUtton 3 has priority

        button_vec = self._get_direction_buttons()
        if sum(button_vec) == 0:  # Nothing
            return 2
        elif button_vec[0] == True:  # Left
            return 3
        elif button_vec[1] == True:  # RIght
            return 4
        else:
            return 5

    def get_recording(self):

        if self._autopilot:
            if self._skiped_frames >= 20:
                return True
            else:
                self._skiped_frames += 1
                return False

        else:
            if (self.joystick.get_button(8)):
                self._recording = True
            if (self.joystick.get_button(9)):
                self._recording = False

            return self._recording

    def get_reset(self):

        if self._autopilot:
            if time.time() - self._start_time > self._reset_period:

                self._reset()
            elif self._latest_measurements.player_measurements.collision_vehicles > 0.0 \
                    or self._latest_measurements.player_measurements.collision_pedestrians > 0.0 or self._latest_measurements.player_measurements.collision_other > 0.0:

                self._reset()


        else:
            if (self.joystick.get_button(4)):
                self._reset()

    def get_noise(self, noise):
        if (self.joystick.get_button(5)):
            if noise == True:
                return False

        return noise

    def get_waypoints(self):

        wp1, wp2 = self._agent.get_active_wps()
        return [wp1, wp2]

    def compute_action(self, sensor, speed):

        """ Get Steering """
        if not self._autopilot:
            steering_axis = self.joystick.get_axis(0)

            acc_axis = self.joystick.get_axis(2)

            brake_axis = self.joystick.get_axis(3)

            if (self.joystick.get_button(3)):
                self._rear = True
            if (self.joystick.get_button(2)):
                self._rear = False

            control = Control()
            control.steer = steering_axis
            control.throttle = -(acc_axis - 1) / 2.0
            control.brake = -(brake_axis - 1) / 2.0
            if control.brake < 0.001:
                control.brake = 0.0

            control.hand_brake = 0
            control.reverse = self._rear


        else:
            control = self._latest_measurements.player_measurements.autopilot_control
            #control.steer += random.uniform(-0.1, 0.1)
            #control = self._agent.run_step(self._latest_measurements,None, self.positions[self.episode_config[1]])

        return control

    def get_sensor_data(self):
        measurements, sensor_data = self.carla.read_data()
        print (sensor_data)
        self._latest_measurements = measurements
        player_data = measurements.player_measurements
        pos = [player_data.transform.location.x, player_data.transform.location.y, 22]
        ori = [player_data.transform.orientation.x, player_data.transform.orientation.y,
               player_data.transform.orientation.z]
        #print measurements
        print sensor_data
        return measurements, sensor_data

        #, direction

    '''
        if self.use_planner:
            if sldist([player_data.transform.location.x, player_data.transform.location.y],
                      [self.positions[self.episode_config[1]].location.x,
                       self.positions[self.episode_config[1]].location.y]) < self._dist_to_activate:
                self._reset()

            #print 'Selected Position ', self.episode_config[1], 'from len ', len(self.positions)
            direction = self.planner.get_next_command(pos, ori, [self.positions[self.episode_config[1]].location.x,
                                                                    self.positions[self.episode_config[1]].location.y,
                                                                    22], (1, 0, 0))
            print direction
        else:
            direction = 2.0
    '''



    def act(self, action):
        self.carla.send_control(action)
