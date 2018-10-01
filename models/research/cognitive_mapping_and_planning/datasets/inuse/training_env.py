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

import pickle
import src.utils as utils

#TODO: action_noise does not use rng




cityfile = 'datasets/inuse/drive_interfaces/carla/carla_client/carla/planner/carla_1.png'
cityfile= 'datasets/inuse/drive_interfaces/carla/carla_client/carla/planner/carla_1.png'
readout_img = Image.open(cityfile)
img_array = np.asarray(readout_img)

readout = False

batch_size = 18

num_weather = 14



sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class ImageData():

    def __init__(self):
        self.raw_rgb = []
        self.rgb = []
        self.depth = []
        self.scene_seg = []


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))

def compute_angle(x0, y0, x1, y1):
    cosv = (x0*x1 + y0*y1)/np.sqrt((x0**2+y0**2)*(x1**2+y1**2))
    #print x0, y0, x1, y1, (x0**2+y0**2)*(x1**2+y1**2), cosv
    return cosv

def load(dataset_path):
    file = open(dataset_path, 'rb')
    dataset = pickle.load(file)
    file.close()
    return dataset

def random_load(rng):
    dataset_base = '/scratch/haoran/inuse/data/carla_collect/noiser_3cam/'
    weather_id = rng.choice(range(num_weather)) + 1
    dataset_path = dataset_base + 'Carla_3Cams_W' + str(weather_id) + '/2018930_Unknown_1/'
    dirs = os.listdir(dataset_path)
    dataset = rng.choice(dirs)
    return load(dataset_path+dataset)



class CarlaEnvMultiplexer:
    def __init__(self, logdir, city_id):
        #self.drive_config = configDrive()
        self.drive_configs = []
        
        self.carla_env_wrapper = None
        self.logdir = logdir
        self.city_id = city_id

    def sample_env(self, rngs):
        #get carla_env_wrapper
        rng = rngs[0]
        if self.carla_env_wrapper == None:
            self.carla_env_wrapper = CarlaEnvWrapper(self.logdir, self.city_id, rng)
        return self.carla_env_wrapper
    



class CarlaEnvWrapper():

    def __init__(self, logdir, city_id, rng):
        self.carla_envs = []
        self.logdir = logdir
        self.map_scales = [0.0025, 0.005, 0.01]
        for i in range(batch_size):
            print ('Environment {:d}'.format(i))
            self.carla_envs.append(CarlaEnv(i, logdir, city_id, rng))
    

    def get_features_name(self):
        f = []
        f.append('imgs')
        if readout:
            for i in range(len(self.map_scales)):
                f.append('readout_maps_{:d}'.format(i))
            for i in range(len(self.map_scales)):
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

    
    def get_action(self):
        """Returns the implemented action from the current node."""
        actions = []
        for i in range(len(self.carla_envs)):
            control = self.carla_envs[i].get_control()
            actions.append(control)
        actions = np.array(actions).astype(np.float32)
        return actions

    def get_targets(self):
        """Returns the optimal action from the current node."""
        optimal_actions = []
        for i in range(len(self.carla_envs)):
            optimal_control = self.carla_envs[i].get_optimal_action()
            optimal_actions.append(optimal_control)
        optimal_actions = np.array(optimal_actions).astype(np.float32)

        return vars(utils.Foo(action=optimal_actions))
    
    def get_targets_name(self):
        return ['action']

    def take_action(self, running_sum_nums):
        #print(running_sum_nums)
        for i in range(len(self.carla_envs)):
            self.carla_envs[i].act(running_sum_nums[i])
        
        
    


class CarlaEnv():

    def __init__(self, env_id, logdir, city_id, rng):
        self.iter = -1
        self.id = env_id
        self.rng = rng
        self.typ = 'rgb'
        
        #self.data = random_load()
        self.len = 0
        self.step_number = 0
        self.need_reset = True
        
        
    def _reset(self):
        print('reset')
        self.saveimgs = False

        self.iter += 1
        
        self.data = random_load(self.rng)

        self.len = len(self.data)

        self.prev_data = None
        self.current_data = None
        
        

        self.running_sum_num = np.zeros((64, 64, 32))
        
        self._start_time = time.time()
        

        self.step_number = 0

        self.need_reset = False
        

    def get_control(self):
        return [self.current_data['data_rewards'][5], self.current_data['data_rewards'][6], self.current_data['data_rewards'][7]]

    def get_optimal_action(self):
        return [self.current_data['data_rewards'][0], self.current_data['data_rewards'][1], self.current_data['data_rewards'][2]]

    def get_data(self):
        if self.len == self.step_number:
            self._reset()
        self.current_data = self.data[self.step_number]
        
        capture_time = time.time()
        if self.typ == 'rgb':
            image = self.current_data['rgb']
        elif self.typ == 'd':
            image = self.current_data['dep']
        else:
            logging.fatal('Sampling not one of uniform.')
        

        data_rewards = self.current_data['data_rewards']


        outs = {}
        
        outs['imgs'] = [self.image_preprocess(image)]
        outs['running_sum_num'] = self.running_sum_num
        outs['command'] = [0.0, 0.0, 0.0, 0.0]
        outs['command'][int(round(data_rewards[25]-2.0))] = 1.0
        outs['speed'] = [data_rewards[11]]

        current_theta = np.arctan2(data_rewards[23], data_rewards[22])
        
        if (self.step_number > 0):
            history_data_rewards = self.prev_data['data_rewards']
            square = np.square(data_rewards[9]-history_data_rewards[9]) +np.square(data_rewards[8]-history_data_rewards[8])
            length = np.sqrt(square)
            if length < 0.1:
                translation_theta = current_theta
            else:
                translation_theta = np.arctan2(data_rewards[9]-history_data_rewards[9], data_rewards[8]-history_data_rewards[8])
            previous_theta = np.arctan2(history_data_rewards[23], history_data_rewards[22])
        else:
            translation_theta = current_theta
            previous_theta = current_theta
            length = 0.0
        outs['incremental_thetas'] = [-(translation_theta - previous_theta), -(current_theta - translation_theta)]
        outs['incremental_locs'] = [0.0, length]

        return outs

    
    
    def act(self, running_sum_num):
        self.step_number += 1
        self.running_sum_num = running_sum_num
        self.prev_data = self.current_data


    def image_preprocess(self, image):
        if self.typ == 'rgb':
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

    '''def get_readout_maps(self, pos):
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
        return readout_maps'''


'''
    
    
        def close(self):
        self.carla.stop()
        print('stop finish')


'''

'''
import random, cv2, time, threading, sys, Queue

import numpy as np
#from joblib import Parallel, delayed
from multiprocessing import Process, Pool
from multiprocessing import Queue as mQueue
import tensorflow as tf
from codification import *
from codification import encode, check_distance

class Dataset(object):
    def __init__(self, splited_keys, images, datasets, config_input, augmenter, perception_interface):
        # sample inputs
        # splited_keys: _splited_keys_train[i_labels_per_division][i_steering_bins_perc][a list of keys]
        # images: [i_sensor][i_file_number] = (lastidx, lastidx + x.shape[0], x)
        # datasets: [i_target_name] = dim*batch matrix, where batch=#all_samples
        # config_input: configInputs
        # augmenter: config_input.augment

        # save the inputs
        self._splited_keys = splited_keys
        self._images = images
        self._targets = np.concatenate(tuple(datasets), axis=1)  # Cat the datasets, The shape is totalnum*totaldim
        self._config = config_input
        self._augmenter = augmenter

        self._batch_size = config_input.batch_size

        # prepare all the placeholders: 3 sources: _queue_image_input, _queue_targets, _queue_inputs
        self._queue_image_input = tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                    config_input.feature_input_size[0],
                                                                    config_input.feature_input_size[1],
                                                                    config_input.feature_input_size[2]])

        self._queue_shapes = [self._queue_image_input.shape]

        # config.targets_names: ['wp1_angle', 'wp2_angle', 'Steer', 'Gas', 'Brake', 'Speed']
        self._queue_targets = []
        for i in range(len(self._config.targets_names)):
            self._queue_targets.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                         self._config.targets_sizes[i]]))
            self._queue_shapes.append(self._queue_targets[-1].shape)

        # self.inputs_names = ['Control', 'Speed']
        self._queue_inputs = []
        for i in range(len(self._config.inputs_names)):
            self._queue_inputs.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                        self._config.inputs_sizes[i]]))
            self._queue_shapes.append(self._queue_inputs[-1].shape)

        self._queue = tf.FIFOQueue(capacity=config_input.queue_capacity,
                                   dtypes=[tf.float32] + [tf.float32] * (len(self._config.targets_names) + len(self._config.inputs_names)),
                                   shapes=self._queue_shapes)
        self._enqueue_op = self._queue.enqueue([self._queue_image_input] + self._queue_targets + self._queue_inputs)
        self._dequeue_op = self._queue.dequeue()

        #self.parallel_workers = Parallel(n_jobs=8, backend="threading")
        self.input_queue = mQueue(5)
        self.output_queue = mQueue(5)

        self.perception_interface = perception_interface


    def get_batch_tensor(self):
        return self._dequeue_op

    def sample_positions_to_train(self, number_of_samples, splited_keys):
        out_splited_keys = []
        for sp in splited_keys:
            if len(sp)>0:
                out_splited_keys.append(sp)

        return np.random.choice(range(len(out_splited_keys)),
                                size=number_of_samples,
                                replace=True), \
               out_splited_keys

    # Used by next_batch, for each of the control block,
    def datagen(self, batch_size, number_control_divisions):
        # typical input: batch_size, number_control_divisions=3, since 3 blocks
        # Goal: uniformly select from different control signals (group), different steering percentiles.
        generated_ids = np.zeros((batch_size, ), dtype='int32')

        count = 0
        to_be_decoded = [[] for _ in range(len(self._images))]
        for control_part in range(0, number_control_divisions):
            num_to_sample = int(batch_size // number_control_divisions)
            if control_part == (number_control_divisions - 1):
                num_to_sample = batch_size - (number_control_divisions - 1) * num_to_sample

            sampled_positions, non_empty_split_keys = self.sample_positions_to_train(num_to_sample,
                                                               self._splited_keys[control_part])

            for outer_n in sampled_positions:
                i = random.choice(non_empty_split_keys[outer_n])
                for isensor in range(len(self._images)):
                    # fetch the image from the h5 files
                    per_h5_len = self._images[isensor][0].shape[0]
                    ibatch = i // per_h5_len
                    iinbatch = i % per_h5_len
                    imencoded = self._images[isensor][ibatch][iinbatch]
                    to_be_decoded[isensor].append(imencoded)

                generated_ids[count] = i
                count += 1

        return to_be_decoded, generated_ids

    """Return the next `batch_size` examples from this data set."""

    # Used by enqueue
    def next_batch(self, sensors, generated_ids):
        # generate unbiased samples;
        # apply augmentation on sensors and segmentation labels
        # normalize images
        # fill in targets and inputs. with reasonable valid condition checking

        batch_size = self._batch_size

        # Get the images -- Perform Augmentation!!!
        for i in range(len(sensors)):
            # decode each of the sensor in parallel
            func = lambda x: cv2.imdecode(x, 1)
            if hasattr(self._config, "hack_resize_image"):
                height, width = self._config.hack_resize_image
                func_previous = func
                func = lambda x: cv2.resize(func_previous(x), (width, height))

            if hasattr(self._config, "hack_faster_aug"):
                func_previous = func
                func = lambda x: func_previous(x)[::2, ::2, :]

            # func = delayed(func)
            # results = self.parallel_workers(func(x) for x in to_be_decoded[isensor])
            results = []
            for x in sensors[i]:
                results.append(func(x))
            sensors[i] = np.stack(results, 0)

            # from bgr to rgb
            sensors[i] = sensors[i][:, :, :, ::-1]

            if self._augmenter[i] != None:
                sensors[i] = self._augmenter[i].augment_images(sensors[i])

            if self._config.image_as_float[i]:
                sensors[i] = sensors[i].astype(np.float32)
            if self._config.sensors_normalize[i]:
                sensors[i] /= 255.0

        # self._targets is the targets variables concatenated
        # Get the targets
        target_selected = self._targets[generated_ids, :]
        target_selected = target_selected.T

        # prepare the output targets, and inputs
        targets = []
        for i in range(len(self._config.targets_names)):
            targets.append(np.zeros((batch_size, self._config.targets_sizes[i])))
        inputs = []
        for i in range(len(self._config.inputs_names)):
            inputs.append(np.zeros((batch_size, self._config.inputs_sizes[i])))

        for ibatch in range(0, batch_size):
            for itarget in range(len(self._config.targets_names)):
                # Yang: This is assuming that all target names has size 1
                k = self._config.variable_names.index(self._config.targets_names[itarget])
                targets[itarget][ibatch] = target_selected[k, ibatch]
                this_name = self._config.targets_names[itarget]

                if this_name == "Speed":
                    # Yang: speed_factor is normalizing the speed
                    targets[itarget][ibatch] /= self._config.speed_factor / 3.6
                elif this_name == "Gas":
                    # Yang: require Gas >=0
                    targets[itarget][ibatch] = max(0, targets[itarget][ibatch])
                elif this_name == "Brake":
                    # Yang: require 0<=Brake<=1
                    targets[itarget][ibatch] = min(1.0, max(0, targets[itarget][ibatch]))

            for iinput in range(len(self._config.inputs_names)):
                k = self._config.variable_names.index(self._config.inputs_names[iinput])
                this_name = self._config.inputs_names[iinput]

                if this_name == "Control":
                    inputs[iinput][ibatch] = encode(target_selected[k, ibatch])
                elif this_name == "Speed":
                    inputs[iinput][ibatch] = target_selected[k, ibatch] / self._config.speed_factor * 3.6
                elif this_name == "Distance":
                    inputs[iinput][ibatch] = check_distance(target_selected[k, ibatch])
                else:
                    raise ValueError()

        # change the output sensors variable
        sensors = np.concatenate(sensors, axis=0)

        return sensors, targets, inputs

    # Used by enqueue
    def process_run(self, sess, data_loaded):
        reshaped = data_loaded[0]
        nB, nH, nW, nC = reshaped.shape
        num_sensors = len(self._config.sensor_names)
        reshaped = np.reshape(reshaped, (num_sensors, nB//num_sensors, nH, nW, nC))
        reshaped = np.transpose(reshaped, (1, 2, 0, 3, 4))
        # now has shape nB//num_sensors, nH, num_sensors, nW, nC
        reshaped = np.reshape(reshaped, (nB//num_sensors, nH, num_sensors*nW, nC))

        if hasattr(self._config, "add_gaussian_noise"):
            std = self._config.add_gaussian_noise
            print("!!!!!!!!!!!!!!!!!!adding gaussian noise", std)
            reshaped += np.random.normal(0, std, reshaped.shape)

        queue_feed_dict = {self._queue_image_input: reshaped}  # images we already put by default

        for i in range(len(self._config.targets_names)):
            queue_feed_dict.update({self._queue_targets[i]: data_loaded[1][i]})

        for i in range(len(self._config.inputs_names)):
            queue_feed_dict.update({self._queue_inputs[i]: data_loaded[2][i]})

        sess.run(self._enqueue_op, feed_dict=queue_feed_dict)

    def __getstate__(self):
        """Return state values to be pickled."""
        print("pickling")
        return (self._splited_keys, self._targets, self._config, self._augmenter, self._batch_size)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        print("unpickling")
        self._splited_keys, self._targets, self._config, self._augmenter, self._batch_size = state

    def _thread_disk_reader(self):
        while True:
            #start = time.time()
            sensors, generated_ids = self.datagen(self._batch_size, len(self._splited_keys))
            self.input_queue.put((sensors, generated_ids))
            #print('putting ele into input queue, cost', time.time()-start)

    @staticmethod
    def _thread_decode_augment(dataset, input_queue, output_queue):
        while True:
            sensors, generated_ids = input_queue.get()
            out = dataset.next_batch(sensors, generated_ids)
            output_queue.put(out)

    def start_multiple_decoders_augmenters(self):
        n_jobs = 6
        for i in range(n_jobs):
            p = Process(target=self._thread_decode_augment, args=(self, self.input_queue, self.output_queue))
            #p = threading.Thread(target=self._thread_decode_augment, args=(self, self.input_queue, self.output_queue))
            p.start()

    def _thread_perception_splitting(self, input_queue):
        while True:
            one_batch = input_queue.get()
            self.output_remaining_queue.put(one_batch[1:])
            self.output_image_queue.put(one_batch[0])

    def _thread_perception_concat(self, perception_output):
        while True:
            remain = self.output_remaining_queue.get()
            image_feature = perception_output.get()
            self.final_output_queue.put([image_feature, remain[0], remain[1]])

    def _thread_feed_dict(self, sess, output_queue):
        while True:
            #start = time.time()
            one_batch = output_queue.get()
            self.process_run(sess, one_batch)
            #print("fetched one output, cost ", time.time()-start)

    def start_all_threads(self, sess):
        t = threading.Thread(target=self._thread_disk_reader)
        t.isDaemon()
        t.start()

        self.start_multiple_decoders_augmenters()

        if self._config.use_perception_stack:
            self.output_image_queue = Queue.Queue(5)
            self.output_remaining_queue = Queue.Queue(5)
            t = threading.Thread(target=self._thread_perception_splitting, args=(self.output_queue,))
            t.start()

            perception_output = self.perception_interface.compute_async_thread_channel(self.output_image_queue)

            self.final_output_queue = Queue.Queue(5)
            t = threading.Thread(target=self._thread_perception_concat, args=(perception_output,))
            t.start()
            output_queue = self.final_output_queue
        else:
            output_queue = self.output_queue

        t = threading.Thread(target=self._thread_feed_dict, args=(sess, output_queue))
        t.isDaemon()
        t.start()


'''