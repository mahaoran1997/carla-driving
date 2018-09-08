#!/usr/bin/env python
import sys

sys.path.append('utils')
sys.path.append('configuration')

import argparse
import numpy as np
import h5py
import pygame
# import readchar
# import json
# from keras.models import

from drawing_tools import *
import time

pygame.init()
network_input_size = [88, 200, 3]

size = (network_input_size[1] * 4, network_input_size[0] * 4)
pygame.display.set_caption("Adas  viewer")

screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

camera_surface = pygame.surface.Surface((network_input_size[1], network_input_size[0]), 0, 24).convert()

# camera_surface_2 = pygame.surface.Surface((network_input_size[1],network_input_size[0]),0,24).convert()

# gta_surface = get_gta_map_surface()


# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--dataset', type=str, default="2016-06-08--11-46-01", help='Dataset/video clip name')
    args = parser.parse_args()

    # config.config_train.batch_size =20
    # config.config_train.is_training=False





    # with open(args.model, 'r') as jfile:
    #  model = model_from_json(json.load(jfile))

    # model.compile("sgd", "mse")
    # weights_file = args.model.replace('json', 'keras')
    # model.load_weights(weights_file)

    # default dataset is the validation data on the highway
    dataset = args.dataset
    first_time = True

    # 297
    # 305
    # 268
    # 351
    # valid_pos_train = #[  34, 46, 28, 0, 58, 33, 14, 40, 54, 8, 59, 7, 31, 11, 62, 66, 64, 57, 63, 67, 78, 5, 29, 42, 1, 41, 2, 10, 77, 75, 56, 53, 47, 6, 15, 74, 71, 9, 61, 79, 52, 72, 25, 3, 44, 26, 49, 51, 60, 73, 13, 37, 43, 76, 32, 65, 35, 50]

    neg_drift = []
    pos_drift = []
    gen_drift = []
    # positions_to_test =
    # positions_to_test = [1003]
    # valid_pos_train = [62, 66, 64, 57, 63, 67, 78, 5, 29, 42, 1, 41, 2, 10, 77, 75, 56, 53, 47, 6, 15, 74, 71, 9, 61, 79, 52, 72, 25, 3, 44, 26, 49, 51, 60, 73, 13, 37, 43, 76, 32, 65, 35, 50]
    count = 0
    steering_pred = []
    steering_gt = []

    # positions_to_test = range(6740,6472) + range(2940,3018) + range(3075,3078) + \
    # range(3198,3201) + range(3874,3876) + range(7579,7581)
    # positions_to_test = [2,3,4,5,6,7]
    # positions_to_test = [30,31,32,33,34,48,49,71,99,155,164,352,325,237,258,228,292]
    positions_to_test = list(range(0, 100))
    # positions_to_test = [93,104,170,173,229,245,283,397,413,425,565,581,591]
    # positions_to_test = range(0,660)
    # positions_to_test = [617,618,619,620,622,623,624,636,637,638,639]
    # positions_to_test =  [637,638]
    # positions_to_test = [55,108,109,353,410,411,426,441,442]
    # positions_to_test = [656,657,675,676,854,855,859,860,861,902]
    path = '/media/nvidia/SSD/GitHub/Desktop/'

    output_file = open('test_yaw.csv', 'a+')
    # data_matrix = np.loadtxt(open(path + "outputsTest1.csv", "rb"), delimiter=",", skiprows=0)
    for h_num in positions_to_test:

        print(" SEQUENCE NUMBER ", h_num)
        data = h5py.File(path + 'data_' + str(h_num).zfill(5) + '.h5', "r")

        # redata = h5py.File('/media/adas/012B4138528FF294/NewGTA/redata_'+ str(h_num).zfill(5) +'.h5', "r")
        # print log.keys()



        # save_data_stats = '../../../Data/Udacity/'





        # skip to highway
        for i in range(200):

            # img = cam['X'][log['cam1_ptr'][i]].swapaxes(0,2).swapaxes(0,1)

            img = np.array(data['images_center'][i])
            direction = 0

            # reimg = np.array(redata['images_center'][i])
            # recontrol_input = np.array(redata['control'][i][1])
            # print img
            # img = img*255
            # print img

            img = img.astype(np.uint8)
            # reimg = reimg.astype(np.uint8)
            # print "sHApE"

            # print "DURATION"
            # print duration
            # print output
            # else:
            #   predicted_steers = output[0][5]*300.0


            # predicted_steers = (output[0][4])*300.0
            # print steer
            # print predicted_steers

            # img_act = get_activation_effects(sess,feedDict,features,weights)
            # img_act = img_act*255
            # img_act = img_act.astype(np.uint8)
            # output_manager.get_activation_effects(feedDict)
            # img_act = scipy.misc.imresize(img_act,(config.input_size[1],config.input_size[0]))


            # print data['targets'][i]

            angle_steers = data['targets'][i][0]
            # print data['targets'][i]
            # print data['targets'][i][6]
            acc = data['targets'][i][1]
            brake = data['targets'][i][2]

            # reangle_steers = redata['targets'][i][0]
            # reacc = redata['targets'][i][1]
            # print data['targets'][i][5], 'b',data['targets'][i][6]
            speed_ms = 10
            # print angle_steers
            # speed_ms = data['targets'][i][2]
            # print angle_steers
            # if abs(angle_steers) > 1.0:
            #  print angle_steers

            # print data['control'][i][0]

            steering_gt.append(angle_steers)

            # img = img[:, :, ::-1]

            # img =

            # print img_act.shape
            draw_path_on(img, speed_ms, -angle_steers * 20.0)
            draw_bar_on(img, acc, img.shape[0] / 8)
            draw_bar_on(img, brake, img.shape[0] / 6, (255, 0, 0))
            # posx = data['targets'][i][7]
            # posy = data['targets'][i][8]
            # draw_path_on(reimg, speed_ms, -reangle_steers*20.0)
            # draw_bar_on(reimg, reacc, reimg.shape[0]/8)
            time.sleep(0.3)
            # draw on
            pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
            # pygame.surfarray.blit_array(camera_surface_2, reimg.swapaxes(0,1))
            # pygame.surfarray.blit_array(activation_surface, img_act)
            myfont = pygame.font.SysFont("monospace", 35)
            camera_surface_2x = pygame.transform.scale2x(camera_surface)
            camera_surface_2x = pygame.transform.scale2x(camera_surface_2x)
            # camera_surface_2_2x = pygame.transform.scale2x(camera_surface_2)
            # activation_surface_2x = pygame.transform.scale2x(activation_surface)
            screen.blit(camera_surface_2x, (0, 0))
            # plot_point(screen,posx,posy,gta_surface)
            # screen.blit(camera_surface_2_2x, (512,0))
            # pygame.display.flip()

            if direction == 4:
                text = "Right"
            elif direction == 3:
                text = "Left"
            elif direction == 5:
                text = "Straight"
            else:
                text = 'Nothing'

            if direction == 2:
                label = myfont.render(text, 1, (255, 0, 0))
            else:
                if (i / 10) % 2 == 0:
                    label = myfont.render(text, 1, (255, 0, 0))
                else:
                    label = myfont.render(text, 1, (0, 255, 0))

            screen.blit(label, (100, 100))

            # if data['targets'][i][22]:
            #  screen.blit( myfont.render("Will Complete", 1, (0,255,0)), (200, 100))
            # else:
            #  screen.blit( myfont.render("Not Completing", 1, (255,0,0)), (200, 100))



            # screen.blit(myfont.render(str(data['targets'][i][10]), 1, (0,255,0)), (100, 200))

            # print goal_pos_x,goal_pos_y


            # dist_tex = myfont.render(str(int(dist)), 1, (255,0,0))
            # screen.blit(dist_tex,(200, 200))
            # screen.blit(activation_surface_2x, (config.input_size[1]*2,0))
            pygame.display.flip()
            # readchar.readchar()
    output_file.close()

    # save_gta_surface(gta_surface)
