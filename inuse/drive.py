import traceback

import sys

sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla')
sys.path.append('drive_interfaces/carla/carla_client')
sys.path.append('drive_interfaces/carla/comercial_cars')

sys.path.append('drive_interfaces/carla/virtual_elektra')
sys.path.append('drive_interfaces/gta_interface')
sys.path.append('drive_interfaces/deeprc_interface')
sys.path.append('drive_interfaces/carla/carla_client/testing')
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('structures')
sys.path.append('evaluation')

import math
import argparse
from noiser import Noiser
import configparser
import datetime

from screen_manager import ScreenManager

import numpy as np
import os
import time

# from config import *
# from eConfig import *
from drawing_tools import *
from extra import *

pygame.init()
clock = pygame.time.Clock()


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


# TODO: TURN this into A FACTORY CLASS
def get_instance(drive_config, experiment_name, drivers_name, memory_use):
    if drive_config.interface == "Carla":

        from carla_recorder import Recorder

        if drive_config.type_of_driver == "Human":
            from carla_human import CarlaHuman
            driver = CarlaHuman(drive_config)


    if hasattr(drive_config, 'carla_config'):
        camera_dict = get_camera_dict(drive_config.carla_config)
        print " Camera Dict "
        print camera_dict

    folder_name = str(datetime.datetime.today().year) + str(datetime.datetime.today().month) + str(
        datetime.datetime.today().day)

    if drivers_name is not None:
        folder_name += '_' + drivers_name
    folder_name += '_' + str(get_latest_file_number(drive_config.path, folder_name))

    if hasattr(drive_config, 'carla_config'):
        print (drive_config.path)
        if not os.path.exists(drive_config.path):
            os.mkdir(drive_config.path)
        recorder = Recorder(drive_config.path + folder_name + '/', drive_config.resolution, \
                            image_cut=drive_config.image_cut, camera_dict=camera_dict, record_waypoints=False)

    return driver, recorder


def drive(experiment_name, drive_config, name=None, memory_use=1.0):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name

    driver, recorder = get_instance(drive_config, experiment_name, name, memory_use)

    noiser = Noiser(drive_config.noise)

    print 'before starting'
    driver.start()

    first_time = True
    new = True

    direction = 2

    iteration = 0
    try:
        while direction != -1:
            capture_time = time.time()
            print(iteration)
            measurements, direction, new2 = driver.get_sensor_data() 
            speed = measurements['PlayerMeasurements'].forward_speed
            
            actions = driver.compute_action()  # measurements.speed
            action_noisy, drifting_time, will_drift = noiser.compute_noise(actions, speed)

            # sensor_data = frame2numpy(image,[800,600])
            recording = driver.get_recording()
            recorder.record(measurements, actions, action_noisy, direction, driver.get_waypoints(), (new or new2), recording)

            iteration += 1
            driver.act(action_noisy)

            new = driver.get_reset()
            

            # print actions
            

            

    except:
        traceback.print_exc()
