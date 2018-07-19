from __future__ import print_function


import pdb
#from scene_parameters import SceneParams

from PIL import Image
import numpy as np
import json, csv

import random
import time
import sys
import argparse
import logging
from socket import error as socket_error
sys.path.append('protoc')


sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla_interface')
sys.path.append('drive_interfaces/carla_interface/carla_client')

sys.path.append('drive_interfaces/carla_interface/carla_client/protoc')
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('input')
sys.path.append('train')
sys.path.append('utils')
sys.path.append('input/spliter')
sys.path.append('structures')


from carla import  SceneDescription,EpisodeStart,EpisodeReady,Control,Measurements
from Manual import *
from carla import CARLA
from carla import Planner

def generate_camera_str(dic):
    camera_str = '''
[CARLA/SceneCapture/%s]
PostProcessing=%s
ImageSizeX=%d
ImageSizeY=%d
CameraFOV=%d
CameraPositionX=%d
CameraPositionY=%d
CameraPositionZ=%d
CameraRotationPitch=%d
CameraRotationRoll=%d
CameraRotationYaw=%d''' % (dic['NAME'], dic['TYPE'], dic['ImageSizeX'], dic['ImageSizeY'], 
    dic['CameraFOV'], dic['CameraPositionX'], dic['CameraPositionY'],
    dic['CameraPositionZ'], dic['CameraRotationPitch'], 
    dic['CameraRotationRoll'], dic['CameraRotationYaw'])
    
    return (dic['NAME'], camera_str)

    
def generate_conffile_from_dict(opt_dict):
    str_preamble = '''
[CARLA/Server]
UseNetworking=true
WorldPort=2000
WritePort=2001
ReadPort=2002'''

    str_level = '''
[CARLA/LevelSettings]
NumberOfVehicles=%d
NumberOfPedestrians=%d
WeatherId=%d''' % (opt_dict['VEHICLES'], opt_dict['PEDESTRIANS'], opt_dict['WEATHER'])
                
                
    str_camera_pre = '''
[CARLA/SceneCapture]
Cameras='''
              
    str_camera_pos = ''
    N = len(opt_dict['CAMERAS'])
    for c in range(N):
        (camera_name, str_camera) = generate_camera_str(opt_dict['CAMERAS'][c])
        str_camera_pos += ('\n' + str_camera)
        
        str_camera_pre += camera_name
        if(c+1 != N):
            str_camera_pre += ','
           
    str_cameras = str_camera_pre + '\n' + str_camera_pos
    str_conf = str_preamble + '\n\n' + str_level + '\n\n' + str_cameras + '\n'
    
    with open(opt_dict['EXP_NAME'], "w") as text_file:
        text_file.write(str_conf)
    

def load_carla_0_parameters():


    # experiment testing conditions
    weathers = [ 1, # clearNoon
                 3, # wetNoon
                 6, # HardRainNoon
                 8, # ClearSunset
            ]
    
    repetitions_per_experiment = [1, 1, 1, 1, 1]
    
    poses_exp1 = [[38, 34],  [4, 2], [12, 10], [62, 55], [43, 47],\
              [64, 66], [78, 76],[59,57],[61,18],[35,39],\
              [12,8],[0,18],[75,68],[54,60],[45,49],\
              [46,42],[53,46],[80,29],[65,63],[0,81],\
              [54,63],[51,42],[16,19],[17,26],[77,68]]
    pedestrians_exp1 = [0] * len(poses_exp1)
    vehicles_exp1 = [0] * len(poses_exp1)
    
    poses_exp2 = [[37, 76], [8, 24], [60, 69], [38, 10], [21, 1],\
               [58,71],[74,32],[44,0],[71,16],[14,24],\
               [34,11],[43,14],[75,16],[80,21],[3,23],\
               [75,59],[50,47],[11,19],[77,34],[79,25] ,\
               [40,63],[58,76],[79,55],[16,61],[27,11]]
    pedestrians_exp2 = [0] * len(poses_exp2)
    vehicles_exp2 = [0] * len(poses_exp2)
    

    poses_exp3 = [[19,66],[79,14],[19,57],[23,1],\
                  [53,76],[42,13],[31,71],[33,5],\
                  [54,30],[10,61],[66,3],[27,12],\
                  [79,19],[2,29],[16,14],[5,57],\
                  [70,73],[46,67],[57,50],[61,49],[21,12],\
                  [51,81],[77,68],[56,65],[43,54]]
    pedestrians_exp3 = [0] * len(poses_exp3)
    vehicles_exp3 = [0] * len(poses_exp3)
    
    poses_exp4 = [[19,66],[79,14],[19,57],[23,1],\
                  [53,76],[42,13],[31,71],[33,5],\
                  [54,30],[10,61],[66,3],[27,12],\
                  [79,19],[2,29],[16,14],[5,57],\
                  [70,73],[46,67],[57,50],[61,49],[21,12],\
                  [51,81],[77,68],[56,65],[43,54]]
    pedestrians_exp4 = [50] * len(poses_exp4)
    vehicles_exp4 = [15] * len(poses_exp4)
    
    poses_exp5 = [[19,66],[79,14],[19,57],[23,1],[53,76],[42,13],[31,71],[33,5],[54,30],[10,61]]
    pedestrians_exp5 = [0] * len(poses_exp5)
    vehicles_exp5 = [0] * len(poses_exp5)
    
    start_goal_poses = [poses_exp1, poses_exp2, poses_exp3, poses_exp4, poses_exp5]
    pedestrians = [pedestrians_exp1, pedestrians_exp2, pedestrians_exp3, pedestrians_exp4, pedestrians_exp5]
    vehicles = [vehicles_exp1, vehicles_exp2, vehicles_exp3, vehicles_exp4, vehicles_exp5]

    return start_goal_poses,pedestrians,vehicles,weathers, repetitions_per_experiment



def load_carla_1_parameters():

    # experiment testing conditions
    weathers = [ 1, # clearNoon
                 3, # wetNoon
                 6, # HardRainNoon
                 8, # ClearSunset
                 # Non Seen in training conditions
                 2, # Cloudy Noon
                 14 # Soft Rain Sunset
            ]
    
    repetitions_per_experiment = [1, 1, 1, 1, 1]
    
    poses_exp1 = [[36,40],[39,35],[110,114],[7,3],[0,4],\
                [68,50],[61,59],[47,64],[147,90],[33,87],\
                [26,19],[80,76],[45,49],[55,44],[29,107],\
                [95,104],[34,84],[51,67],[22,17],[91,148],\
                [20,107],[78,70],[95,102],[68,44],[45,69]]

    pedestrians_exp1 = [0] * len(poses_exp1)
    
    vehicles_exp1 = [0] * len(poses_exp1)
    
    poses_exp2 = [[138,17],[46,16],[26,9],[42,49],[140,26],\
                [85,97],[65,133],[137,51],[76,66],[46,39],\
                [40,60],[1,28],[4,129],[121,107],[2,129],\
                [78,44],[68,85],[41,102],[95,70],[68,129],\
                [84,69],[47,79],[110,15],[130,17],[0,17]]
    pedestrians_exp2 = [0] * len(poses_exp2)
    
    vehicles_exp2 = [0] * len(poses_exp2)

    poses_exp3 = [[105,29],[27,130],[102,87],[132,27],[24,44],\
               [96,26],[34,67],[28,1],[140,134],[105,9],\
               [148,129],[65,18],[21,16],[147,97],[42,51],\
               [30,41],[18,107],[69,45],[102,95],[18,145],\
               [111,64],[79,45],[84,69],[73,31],[37,81]]

    pedestrians_exp3 = [0] * len(poses_exp3)
    vehicles_exp3 = [0] * len(poses_exp3)
    
    poses_exp4 = [[105,29],[27,130],[102,87],[132,27],[24,44],\
               [96,26],[34,67],[28,1],[140,134],[105,9],\
               [148,129],[65,18],[21,16],[147,97],[42,51],\
               [30,41],[18,107],[69,45],[102,95],[18,145],\
               [111,64],[79,45],[84,69],[73,31],[37,81]]
    pedestrians_exp4 = [50] * len(poses_exp4)
    vehicles_exp4 = [20] * len(poses_exp4)

    poses_exp5 = [[105,29],[27,130],[102,87],[132,27],[24,44],[96,26],[34,67],[28,1],[140,134],[105,9]]
    pedestrians_exp5 = [0] * len(poses_exp5)
    vehicles_exp5 = [0] * len(poses_exp5)    
    
    start_goal_poses = [poses_exp1, poses_exp2, poses_exp3, poses_exp4, poses_exp5]
    pedestrians = [pedestrians_exp1, pedestrians_exp2, pedestrians_exp3, pedestrians_exp4, pedestrians_exp5]
    vehicles = [vehicles_exp1, vehicles_exp2, vehicles_exp3, vehicles_exp4, vehicles_exp5]

    return start_goal_poses,pedestrians,vehicles,weathers, repetitions_per_experiment





def run_experiment(opt_dict,city_name):
    carla = opt_dict['CARLA']
    width  = opt_dict['WIDTH']
    height = opt_dict['HEIGHT']
    host = opt_dict['HOST']
    port = opt_dict['PORT']

    output_summary = opt_dict['OUTPUT_SUMMARY']
    runnable = opt_dict['RUNNABLE']
    experiments_to_run = opt_dict['EXPERIMENTS_TO_RUN']
    pedestrians = opt_dict['PEDESTRIANS']
    vehicles = opt_dict['VEHICLES']
    repetitions_per_experiment = opt_dict['REPETITIONS']
    weathers = opt_dict['WEATHERS']
    start_goal_poses = opt_dict['START_GOAL_POSES']
    cameras = opt_dict['CAMERAS']

    list_stats = []
    dict_stats = {'exp_id':-1,
                  'rep':-1,
                  'weather':-1,
                  'start_point':-1,
                  'end_point':-1,
                  'result':-1,
                  'initial_distance':-1,
                  'final_distance':-1,
                  'final_time':-1,

                 }

    dict_rewards = {
                  'exp_id':-1,
                  'rep':-1,
                  'weather':-1,
                  'collision_gen':-1,
                  'collision_ped':-1,
                  'collision_car':-1,
                  'lane_intersect':-1,
                  'sidewalk_intersect':-1,
                  'pos_x':-1,
                  'pos_y':-1
    }
    import os
    dir_path = os.path.dirname(__file__)
    

    planner = Planner(dir_path+'/../carla/planner/' + city_name + '.txt',\
      dir_path+'/../carla/planner/' + city_name + '.png')



    with open(output_summary, 'wb') as ofd:

        with open(output_summary + '_rewards_file','wb' ) as rfd:

            w = csv.DictWriter(ofd, dict_stats.keys())
            w.writeheader()
            rw = csv.DictWriter(rfd, dict_rewards.keys())
            rw.writeheader()


            for experiment_id in experiments_to_run:
                poses_exp = start_goal_poses[experiment_id]
                repetitions = repetitions_per_experiment[experiment_id]
                pedestrians_exp = pedestrians[experiment_id]
                vehicles_exp = vehicles[experiment_id]

                # several repetitions
                for rep in range(repetitions):
                    # for the different weathers
                    for weather_cond in weathers:
                        # let's go through all the starting-goal positions of the experiment

                        for i in range(len(poses_exp)):
                            trajectory = poses_exp[i]
                            ped = pedestrians_exp[i]
                            vehic = vehicles_exp[i]
                            start_point = trajectory[0]
                            end_point = trajectory[1]

                            # generate conf file for these conditions
                            exp_name = "expID_%d_rep_%d_weatherCond_%d__startPoint_%d__endPoint_%d" % (experiment_id,
                                        rep, weather_cond, start_point, end_point)
                            file_exp_name = './temp/' + exp_name + '.txt'
                            settings = {'EXP_NAME':file_exp_name, 'CAMERAS':cameras,
                                        'WEATHER':weather_cond, 'PEDESTRIANS':ped, 'VEHICLES':vehic}
                            generate_conffile_from_dict(settings)

                            if(carla == None):
                                carla = CARLA(host, port)


                            positions = carla.loadConfigurationFile(file_exp_name)
                            carla.newEpisode(start_point)
                            # We compute the total distance for this episode


                            _,path_distance=planner.get_next_command([positions[start_point].location.x\
                            ,positions[start_point].location.y,22],[positions[start_point].orientation.x\
                            ,positions[start_point].orientation.y,22],[positions[end_point].location.x\
                            ,positions[end_point].location.y,22],(1,0,0))
                            # We calculate the timout based on the distance

                            #time_out = ((path_distance/100000.0)/10.0)*3600.0 + 10.0
			    time_out = ((path_distance/100000.0))*3600.0/15 + 10.0


                            # running the agent
                            print('======== !!!! ==========')
                            (result, reward_vec, final_time,distance) = runnable.run_until(carla,time_out , positions[end_point])

                            # compute stats for the experiment
                            dict_stats['exp_id'] = experiment_id
                            dict_stats['rep'] = rep
                            dict_stats['weather'] = weather_cond
                            dict_stats['start_point'] = start_point
                            dict_stats['end_point'] = end_point
                            dict_stats['result'] = result
                            dict_stats['initial_distance'] = distance
                            dict_stats['final_distance'] = distance
                            dict_stats['final_time'] = final_time

                            for i in range(len(reward_vec)):
                                dict_rewards['exp_id'] = experiment_id
                                dict_rewards['rep'] = rep
                                dict_rewards['weather'] = weather_cond
                                dict_rewards['collision_gen'] = reward_vec[i].collision_other
                                dict_rewards['collision_ped'] = reward_vec[i].collision_pedestrians
                                dict_rewards['collision_car'] = reward_vec[i].collision_vehicles
                                dict_rewards['lane_intersect'] = reward_vec[i].intersection_otherlane
                                dict_rewards['sidewalk_intersect'] = reward_vec[i].intersection_offroad
                                dict_rewards['pos_x'] = reward_vec[i].transform.location.x
                                dict_rewards['pos_y'] = reward_vec[i].transform.location.y


                                rw.writerow(dict_rewards)


                            # save results of the experiment
                            list_stats.append(dict_stats)
                            print (dict_stats)
                            w.writerow(dict_stats)


                            if(result > 0):
                                print('+++++ Target achieved in %f seconds! +++++' % final_time)
                            else:
                                print('----- Timeout! -----')
    return list_stats

def main(runnable):
    host = '158.109.9.226'
    port = 2000

    summary_file = 'summary_experiments_00.csv'
    
    # cameras setup
    width = 800
    height = 600
    fov = 100
    c_x = 170
    c_y = 0
    c_z = 150
    r_p = 0
    r_r = 0
    r_y = 0
    
    cameras = [ {'NAME':'RGB', 'TYPE':'SceneFinal', 'ImageSizeX':width, 
                 'ImageSizeY':height, 'CameraFOV':fov, 'CameraPositionX':c_x,
                 'CameraPositionY':c_y, 'CameraPositionZ':c_z,
                 'CameraRotationPitch':r_p, 'CameraRotationRoll':r_r,
                 'CameraRotationYaw':r_y},
    
               {'NAME':'Depth', 'TYPE':'Depth', 'ImageSizeX':width, 
                 'ImageSizeY':height, 'CameraFOV':fov, 'CameraPositionX':c_x,
                 'CameraPositionY':c_y, 'CameraPositionZ':c_z,
                 'CameraRotationPitch':r_p, 'CameraRotationRoll':r_r,
                 'CameraRotationYaw':r_y},
                
               {'NAME':'SEG', 'TYPE':'SemanticSegmentation', 'ImageSizeX':width, 
                 'ImageSizeY':height, 'CameraFOV':fov, 'CameraPositionX':c_x,
                 'CameraPositionY':c_y, 'CameraPositionZ':c_z,
                 'CameraRotationPitch':r_p, 'CameraRotationRoll':r_r,
                 'CameraRotationYaw':r_y}
               ]



    experiments_to_run = [0, 1, 2]
    

    start_goal_poses,pedestrians,vehicles,weathers, repetitions_per_experiment = load_carla_0_parameters()


    timeout = [60,90,150,150]
    opt_dict = {'CARLA':None, 'HOST':host, 'PORT':port, 'TIMEOUT':timeout, 
                'OUTPUT_SUMMARY':summary_file, 'RUNNABLE':runnable, 
                'CAMERAS':cameras, 'WIDTH':width, 'HEIGHT':height, 
                'WEATHERS':weathers, 'REPETITIONS':repetitions_per_experiment,
                'START_GOAL_POSES':start_goal_poses, 'PEDESTRIANS':pedestrians,
                'VEHICLES':vehicles, 'EXPERIMENTS_TO_RUN':experiments_to_run}

    # run experiments
    run_experiment(opt_dict)
            

if(__name__ == '__main__'):

    # instance your controller here
    runnable = Manual()
    main(runnable)
