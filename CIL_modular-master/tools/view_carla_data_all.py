#!/usr/bin/env python
import sys

sys.path.append('utils')
sys.path.append('configuration')

import numpy as np
import h5py
# import readchar
# import json
# from keras.models import

import matplotlib.pyplot as plt
import math
from drawing_tools import *
import time
from collections import deque
import seaborn as sns

sns.set(color_codes=True)

sys.path.append('drive_interfaces')

from screen_manager import ScreenManager


class Control:
    steer = 0
    throttle = 0
    brake = 0
    hand_brake = 0
    reverse = 0


# Configurations for this script

sensors = {'RGB': 3, 'labels': 3, 'depth': 3}
resolution = [200, 88]
camera_id_position = 25
direction_position = 24
speed_position = 10
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


def join_classes(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for key, value in join_dic.items():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


def join_classes_for(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    # print compressed_labels_image.shape
    for i in range(labels_image.shape[0]):
        for j in range(labels_image.shape[1]):
            compressed_labels_image[i, j, 0] = join_dic[labels_image[i, j, 0]]

    return compressed_labels_image


# ***** main loop *****
if __name__ == "__main__":

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []
    path = '/media/matthias/7E0CF8640CF818BB/Github/Desktop/RSS_W1_w1000_2h_WP/'
    step_size = 1
    showSlow = False
    h5start = 587
    h5end = 1080  # len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    # initial_positions =[20,25,48,68,79,105,108,120,130]
    # positions_to_test = []
    # for i in initial_positions:
    #  positions_to_test += range(i-1,i+2)

    positions_to_test = list(range(h5start, h5end + 1))

    screen = ScreenManager()

    image_queue = deque()

    actions_queue = deque()

    # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful

    screen.start_screen([resolution[0] * 2, resolution[1] * 2], [sensors['RGB'], 2], 1)

    ts = []
    images = [np.zeros([resolution[1], resolution[0], 3])] * sensors['RGB']
    labels = [np.zeros([resolution[1], resolution[0], 1])] * sensors['labels']
    depths = [np.zeros([resolution[1], resolution[0], 3])] * sensors['depth']
    actions = [Control()] * sensors['RGB']
    actions_noise = [Control()] * sensors['RGB']

    for h_num in positions_to_test:

        print(" SEQUENCE NUMBER ", h_num)
        try:
            data = h5py.File(path + 'data_' + str(h_num).zfill(5) + '.h5', "r")
        except Exception as e:
            print(e)
            continue

        for i in range(0, 197, sensors['RGB'] * step_size):

            if (showSlow):
                time.sleep(0.25)
            speed = math.fabs(data['targets'][i + 2][speed_position])

            for j in range(sensors['RGB']):
                capture_time = time.time()
                images[int(data['targets'][i + j][camera_id_position])] = np.array(data['rgb'][i + j]).astype(np.uint8)
                # print ' Read RGB time ',time.time() - capture_time
                # depths[int(data['targets'][i +j][25])] = np.array(data['depth'][i+j]).astype(np.uint8)
                action = Control()
                angle = data['targets'][i + j][26]

                #########Augmentation!!!!
                # time_use =  1.0
                # car_lenght = 6.0
                # targets[count][i] -=min(4*(math.atan((angle*car_lenght)/(time_use*float_data[speed_pos,i]+0.05)))/3.1415,0.2)

                action.steer = data['targets'][i + j][0]
                steering_pred.append(action.steer)
                action.throttle = data['targets'][i + j][1]
                action.brake = data['targets'][i + j][2]
                time_use = 1.0
                car_lenght = 6.0
                if angle > 0.0:
                    angle = math.radians(math.fabs(angle))
                    action.steer -= min(6 * (math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / 3.1415, 0.3)
                else:
                    angle = math.radians(math.fabs(angle))
                    action.steer += min(6 * (math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / 3.1415, 0.3)

                # print 'Angle : ',angle,'Steer : ',action.steer


                actions[int(data['targets'][i + j][camera_id_position])] = action

                action_noise = Control()
                action_noise.steer = data['targets'][i + j][0]
                action_noise.throttle = data['targets'][i + j][1]
                action_noise.brake = data['targets'][i + j][2]

                actions_noise[int(data['targets'][i + j][camera_id_position])] = action_noise

            for j in range(sensors['labels']):
                capture_time = time.time()
                labels[int(data['targets'][i + j][camera_id_position])] = np.array(data['labels'][i + j]).astype(
                    np.uint8)
                # print ' Read Label time ',time.time() - capture_time
            for j in range(sensors['depth']):
                depths[int(data['targets'][i + j][camera_id_position])] = np.array(data['depth'][i + j]).astype(
                    np.uint8)

            direction = data['targets'][i][direction_position]
            # print direction
            print()
            speed = data['targets'][i + 2][speed_position]
            # print '[ ',data['targets'][i][27],',',data['targets'][i][28],'],[',data['targets'][i][29],',',data['targets'][i][30],']'


            # actions[0].steer +=min(4*(math.atan((0.26*car_lenght)/(time_use*speed+0.05)))/3.1415,0.2)
            # actions[2].steer -=min(4*(math.atan((0.26*car_lenght)/(time_use*speed+0.05)))/3.1415,0.2)
            # print 'Steer ',actions[0].steer
            # print 'Throttle ',actions[0].throttle
            # print 'Brake ',actions[0].brake

            for j in range(sensors['RGB']):
                screen.plot_camera_steer(images[j], actions[j].steer, [j, 0])
                # print 'images ok'


                # for j in range(sensors['labels']):

                # print labels
                # print j
                # labels[j] =labels[j]*int(255/(number_of_seg_classes-1))
                screen.plot_camera(labels[j], [j, 1])

                # for j in range(sensors['depth']):
                #  #print j

                #  screen.plot_camera(depths[j] ,[j,2])



                # pygame.display.flip()
                # time.sleep(0.05)

    plt.plot(steering_pred)
    plt.show()
    # save_gta_surface(gta_surface)
