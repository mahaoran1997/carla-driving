import traceback

import sys

sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/comercial_cars')
sys.path.append('drive_interfaces/carla_client')
sys.path.append('drive_interfaces/carla_client/PythonClient')

'''
sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla')
sys.path.append('drive_interfaces/carla/carla_client')
sys.path.append('drive_interfaces/carla/comercial_cars')

sys.path.append('drive_interfaces/carla/carla_client/PythonClient')
sys.path.append('drive_interfaces/carla/carla_client/PythonClient/carla')

sys.path.append('drive_interfaces/carla/carla_client/PythonClient/carla/planner')

sys.path.append('drive_interfaces/carla/carla_client/PythonClient/carla/testing')

sys.path.append('drive_interfaces/carla/virtual_elektra')
sys.path.append('drive_interfaces/gta_interface')
sys.path.append('drive_interfaces/deeprc_interface')
sys.path.append('drive_interfaces/carla/carla_client/testing')da
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('structures')
sys.path.append('evaluation')

'''

# from drive_interfaces.carla_client.PythonClient.carla import sensor
from carla import sensor
from carla.client import CarlaClient, make_carla_client
from carla.settings import CarlaSettings
from carla.sensor import Camera, Lidar
import math
import argparse

from noiser import Noiser  # ?
import configparser  # ?
import datetime  # ?

from screen_manager import ScreenManager  # ?

import numpy as np
import os
import time  # ?

# from drive_interfaces.carla_client.PythonClient.carla import image_converter
from carla import image_converter  # ?

from drawing_tools import *  # ?
from extra import *  # ?
from ConfigParser import ConfigParser
import random

pygame.init()
clock = pygame.time.Clock()


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))


def get_camera_dict(ini_file):
    print (ini_file)
    config = configparser.ConfigParser()
    config.read(ini_file)
    cameras = config['CARLA/SceneCapture']['Cameras']
    camera_dict = {}
    cameras = cameras.split(',')
    #print cameras
    for i in range(len(cameras)):
        angle = config['CARLA/SceneCapture/' + cameras[i]]['CameraRotationYaw']
        camera_dict.update({i: (cameras[i], angle)})

    return camera_dict


# TODO: TURN this into A FACTORY CLASS
def get_instance(drive_config, drivers_name, memory_use):
    if drive_config.interface == "Carla":
        from carla_recorder import Recorder
        from carla_human import CarlaHuman
        driver = CarlaHuman(drive_config)
    else:
        print " Not valid interface is set "

    camera_dict = get_camera_dict(drive_config.carla_config)
    print " Camera Dict "
    #print camera_dict

    folder_name = str(datetime.datetime.today().year) + str(datetime.datetime.today().month) + str(
        datetime.datetime.today().day)

    if drivers_name is not None:
        folder_name += '_' + drivers_name
    folder_name += '_' + str(get_latest_file_number(drive_config.path, folder_name))
    print (drive_config.path)
    print (folder_name)
    recorder = Recorder(drive_config.path + folder_name + '/', drive_config.resolution, image_cut=drive_config.image_cut, camera_dict=camera_dict, record_waypoints = False)# record_waypoints=True)

    return driver, recorder

'''
        if drive_config.type_of_driver == "Human":
'''


'''        else:
            from carla_machine import CarlaMachine
            driver = CarlaMachine("0", experiment_name, drive_config, memory_use)

    if drive_config.interface == "VirtualElektra":

        from carla_recorder import Recorder

        if drive_config.type_of_driver == "Human":
            from virtual_elektra_human import VirtualElektraHuman
            driver = VirtualElektraHuman(drive_config)
        else:
            from virtual_elektra_machine import VirtualElektraMachine
            driver = VirtualElektraMachine("0", experiment_name, drive_config, memory_use)
'''

def drive(drive_config, name=None, memory_use=1.0):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name

    driver, recorder = get_instance(drive_config, name, memory_use)

    noiser = Noiser(drive_config.noise)

    print 'before starting'
    with make_carla_client(drive_config.host, drive_config.port) as client:
        config = ConfigParser()
        config.optionxform = str
        config.read(drive_config.carla_config)
        config.set('CARLA/LevelSettings', 'NumberOfVehicles', drive_config.cars)

        config.set('CARLA/LevelSettings', 'NumberOfPedestrians', drive_config.pedestrians)

        config.set('CARLA/LevelSettings', 'WeatherId', drive_config.weather)

        # Write down a temporary init_file to be used on the experiments
        temp_f_name = 'p' + str(drive_config.pedestrians) + '_c' + str(drive_config.cars) + "_w" \
                      + str(drive_config.weather) + '.ini'

        print (temp_f_name)
        print (config)

        with open(temp_f_name, 'w') as configfile:
            config.write(configfile)

        _start_time = time.time() #???

        # print (self._config_path)

        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=random.choice([1, 3, 7, 8, 14]),
            QualityLevel='Epic')
        settings.randomize_seeds()

        # Now we want to add a couple of cameras to the player vehicle.
        # We will collect the images produced by these cameras every
        # frame.

        # The default camera captures RGB images of the scene.
        camera0 = Camera('CameraRGB')
        # Set image resolution in pixels.
        camera0.set_image_size(800, 600)
        # Set its position relative to the car in meters.
        camera0.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera0)

        # Let's add another camera producing ground-truth depth.
        camera1 = Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        camera1.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera1)

        camera = Camera('MyCamera', PostProcessing='SemanticSegmentation')
        camera.set(FOV=90.0)
        camera.set_image_size(800, 600)
        camera.set_position(x=0.30, y=0, z=1.30)
        camera.set_rotation(pitch=0, yaw=0, roll=0)

        settings.add_sensor(camera)

        with open(drive_config.carla_config) as file:
            p = client.load_settings(settings)
            # print (p)
            positions = p.player_start_spots
            number_of_player_starts = len(p.player_start_spots)
        # print (self.positions[0])
        episode_config = [random.randint(0, max(0, number_of_player_starts - 1)),
                               random.randint(0, max(0, number_of_player_starts - 1))]
        # print ('after episode_config')
        # self._agent = Autopilot(ConfigAutopilot(self._driver_conf.city_name))
        # print ('before starting episode')
        client.start_episode(episode_config[0])
        # print ('after starting episode')
        print 'RESET ON POSITION ', episode_config[0]
        #measurements, sensor_data = client.read_data()
        #print (123)
        #print (sensor_data)
        _dist_to_activate = 300
        # Set 10 frames to skip
        _skiped_frames = 0


        first_time = True #????
        direction = 2

        iteration = 0
        time_iteration = 0

        while direction != -1:
            capture_time = time.time()
            measurements, sensor_data = client.read_data()  # Later it would return more image like [rewards,images,segmentation]
            _latest_measurements = measurements
            direction = 1

            # sensor_data = frame2numpy(image,[800,600])
            #print ("Go")
            print (iteration)

            #recording = driver.get_recording()

            if _skiped_frames >= 20:
                recording = True
            else:
                _skiped_frames += 1
                recording = False

            #driver.get_reset()
            # time.time() - _start_time > drive_config.reset_period)
            if (time_iteration > drive_config.reset_period) \
                    or (_latest_measurements.player_measurements.collision_vehicles > 0.0 or _latest_measurements.player_measurements.collision_pedestrians > 0.0 or _latest_measurements.player_measurements.collision_other > 0.0):


                _start_time = time.time()  # ???
                time_iteration = 0
                # print (self._config_path)

                with open(drive_config.carla_config) as file:
                    p = client.load_settings(settings)
                    # print (p)
                    positions = p.player_start_spots
                    number_of_player_starts = len(p.player_start_spots)
                # print (self.positions[0])
                episode_config = [random.randint(0, max(0, number_of_player_starts - 1)),
                                  random.randint(0, max(0, number_of_player_starts - 1))]
                # print ('after episode_config')
                # self._agent = Autopilot(ConfigAutopilot(self._driver_conf.city_name))
                # print ('before starting episode')
                client.start_episode(episode_config[0])
                # print ('after starting episode')
                print 'RESET ON POSITION ', episode_config[0]
                #measurements, sensor_data = client.read_data()
                #print (123)
                #print (sensor_data)
                _dist_to_activate = 300
                # Set 10 frames to skip
                _skiped_frames = 0
                continue

            speed = measurements.player_measurements.forward_speed
            #print sensor_data
            rgbs = [x for name, x in sensor_data.items() if isinstance(x, sensor.Image) and x.type == 'SceneFinal']

            labels = [x for name, x in sensor_data.items() if
                      isinstance(x, sensor.Image) and x.type == 'SemanticSegmentation']

            #print len(rgbs)
            #print len(labels)

            # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed,\
            # driver.compute_direction((measurements.transform.location.x,measurements.transform.location.y,22),\
            # (measurements.transform.orientation.x,measurements.transform.orientation.y,measurements.transform.orientation.z))) #rewards.speed
            # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed) #rewards.speed
            #actions = driver.compute_action([image_converter.to_rgb_array(rgbs[0]), image_converter.to_rgb_array(labels[0])],speed)  # rewards.speed
            actions = measurements.player_measurements.autopilot_control

            action_noisy, drifting_time, will_drift = noiser.compute_noise(actions, speed)

            # print actions
            if recording:
                recorder.record(measurements, sensor_data, actions, action_noisy, direction)#, driver.get_waypoints())


            '''if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:
                image_vbp = driver.compute_perception_activations(measurements['BGRA'][drive_config.middle_camera],
                                                                  speed)

                screen_manager.plot_camera(image_vbp, [1, 0])'''

            iteration += 1
            time_iteration += 1
            #driver.act(action_noisy)
            client.send_control(action_noisy)




















    '''driver.start()
    first_time = True
    if drive_config.show_screen:
        screen_manager = ScreenManager()

        #image = image[self._image_cut[0]:self._image_cut[1],:,:]
        screen_manager.start_screen([800, 600], drive_config.aspect_ratio, drive_config.scale_factor)

    direction = 2

    iteration = 0
    try:
        while direction != -1:
            capture_time = time.time()
            measurements, sensor_data = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]
            direction = 1

            #sensor_data = frame2numpy(image,[800,600])

            # Compute now the direction
            if drive_config.show_screen:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done = True  # Flag that we are done so we exit this loop

            recording = driver.get_recording()
            driver.get_reset()
            speed = measurements.player_measurements.forward_speed
            print sensor_data
            rgbs = [x for name, x in sensor_data.items() if isinstance(x, sensor.Image) and x.type == 'SceneFinal']

            labels = [x for name, x in sensor_data.items() if
                      isinstance(x, sensor.Image) and x.type == 'SemanticSegmentation']

            print len(rgbs)
            print len(labels)

            # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed,\
            # driver.compute_direction((measurements.transform.location.x,measurements.transform.location.y,22),\
            # (measurements.transform.orientation.x,measurements.transform.orientation.y,measurements.transform.orientation.z))) #rewards.speed
            # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed) #rewards.speed
            actions = driver.compute_action(
                [image_converter.to_rgb_array(rgbs[0]), image_converter.to_rgb_array(labels[0])],
                speed)  # rewards.speed

            action_noisy, drifting_time, will_drift = noiser.compute_noise(actions, speed)

            # print actions
            if recording:
                recorder.record(measurements, sensor_data, actions, action_noisy, direction, driver.get_waypoints())

            if drive_config.show_screen:
                if drive_config.interface == "Carla" or drive_config.interface == "VirtualElektra":
                    # for i in range(drive_config.aspect_ratio[0]*drive_config.aspect_ratio[1]):
                    print 'fps', 1.0 / (time.time() - capture_time)
                    # print measurements['BGRA'][drive_config.middle_camera].shape
                    image = image_converter.to_rgb_array(labels[0])
                    image = image[:, :, 0] * 30
                    image = image[:, :, np.newaxis]
                    # print image.shape
                    # image = image[:, :, ::-1]
                    image.setflags(write=1)
                    screen_manager.plot_camera_steer(image, actions.steer, [0, 0])
                    # print 'fps',1.0/(time.time() - capture_time)
                    # print measurements['BGRA'][drive_config.middle_camera].shape
                    image = image_converter.to_rgb_array(rgbs[0])

                    # print image.shape
                    # image = image[:, :, ::-1]
                    image.setflags(write=1)
                    screen_manager.plot_camera_steer(image, actions.steer, [1, 0])

                # mid_rep = mid_rep*255
                # print mid_rep.shape
                # print mid_rep

                # screen_manager.plot_camera_steer(mid_rep,actions.steer,[2,0])

                else:
                    print "Not supported interface"
                    pass

            if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:
                image_vbp = driver.compute_perception_activations(measurements['BGRA'][drive_config.middle_camera],
                                                                  speed)

                screen_manager.plot_camera(image_vbp, [1, 0])

            iteration += 1
            driver.act(action_noisy)

    except:
        traceback.print_exc()

    finally:

        # driver.write_performance_file(path,folder_name,iteration)
        pygame.quit()'''

