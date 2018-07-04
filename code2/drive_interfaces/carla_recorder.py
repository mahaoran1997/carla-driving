import h5py
import scipy
import time
from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from PIL import Image
import numpy as np
import os
import math as m
from carla import sensor
from carla.image_converter import *
import pickle


# lets put a big queue for the disk. So I keep it real time while the disk is writing stuff
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread

    return wrapper


class Recorder(object):

    # We assume a three camera case not many cameras per input ....

    def __init__(self, file_prefix, resolution=[800, 600], current_file_number=0, \
                 record_image=True, image_cut=[0, 600], camera_dict={}, record_waypoints=True):

        self._number_of_seg_classes = 13
        self._record_image_hdf5 = True
        self._number_images_per_file = 200
        self._file_prefix = file_prefix
        self._image_size2 = resolution[1]
        self._image_size1 = resolution[0]
        self._record_image = False
        self._number_rewards = 30 + 4 * record_waypoints
        self._image_cut = image_cut
        self._camera_dict = camera_dict
        self._record_waypoints = record_waypoints
        if not os.path.exists(self._file_prefix):
            os.mkdir(self._file_prefix)

        self._current_file_number = current_file_number

        self._current_hf = self._create_new_db()
        self._current_np = open(self._file_prefix + 'data_' + str(self._current_file_number).zfill(5) + '.pkl', 'wb')
        self._images_writing_folder_vec = []
        # for i in range(max(len(self._camera_dict),1)):
        #	images_writing_folder = self._file_prefix + "Images_" +str(i) + "/"
        #	if not os.path.exists(images_writing_folder):
        #		os.mkdir(images_writing_folder)
        #	self._images_writing_folder_vec.append(images_writing_folder)

        self._csv_writing_file = self._file_prefix + 'outputs.csv'
        self._current_pos_on_file = 0
        self._data_queue = Queue(5000)
        self.run_disk_writer()

    def _create_new_db(self):

        hf = h5py.File(self._file_prefix + 'data_' + str(self._current_file_number).zfill(5) + '.h5', 'w')
        #np =
        # print self._number_images_per_file, self._image_size2, self._image_size1
        self.data_center = hf.create_dataset('rgb',
                                             (self._number_images_per_file, self._image_size2, self._image_size1, 3),
                                             dtype=np.uint8)
        self.segs_center = hf.create_dataset('labels',
                                             (self._number_images_per_file, self._image_size2, self._image_size1, 1),
                                             dtype=np.uint8)
        self.depth_center = hf.create_dataset('depth',
                                              (self._number_images_per_file, self._image_size2, self._image_size1, 1),
                                              dtype=np.uint8)
        '''
        self.vehicles = hf.create_dataset('vehicles', (self._number_images_per_file, self._number_vehicles, 4), 'f')
        self.padestrians = hf.create_dataset('pedestrians', (self._number_images_per_file, self._number_vehicles, 4), 'f')
        self.trafficlight = hf.create_dataset('trafficlight', (self._number_images_per_file, self._number_lights, 4), 'f')
        self.sign = hf.create_dataset('signs', (self._number_images_per_file, self._number_signs, 2), 'f')
        '''
        self.nonplayer_agents_info = []
        self.data_rewards = hf.create_dataset('targets', (self._number_images_per_file, self._number_rewards), 'f')
        return hf  # , np

    def record(self, measurements, action, action_noise, direction, waypoints=None):

        self._data_queue.put([measurements, action, action_noise, direction, waypoints])
        #self._write_to_disk([measurements, action, action_noise, direction, waypoints])

    def get_one_hot_encoding(self, array, numLabels):
        mask = np.zeros((array.shape[0], array.shape[1], numLabels))
        for x in range(len(array)):
            row = array[x]
            for y in range(len(row)):
                label = row[y]
                mask[x, y, label] = 1
        return mask

    @threaded
    def run_disk_writer(self):
        # fo = open("writeBegin.txt", "w")
        # fo.write("writeBegin!")
        # fo.close()

        while True:
            data = self._data_queue.get()
            if self._data_queue.qsize() % 100 == 0:
                print "QSIZE:", self._data_queue.qsize()
            self._write_to_disk(data)

    def _write_to_disk(self, data):
        # Use the dictionary for this
        # fo = open("writeEnd.txt", "w")

        measurements = data[0]
        sensor_data = data[1]
        actions = data[2]
        action_noise = data[3]
        direction = data[4]
        if self._record_waypoints:
            waypoints = data[5]

        # fo.write("data:")
        # fo.write(measurements)
        # for i in range(max(len(rgbs), len(labels), len(depths))):
        # l = max(len(sensor_data['Labels']), len(measurements.depth))#len(measurements.BGRA),
        rgbs = [x for name, x in sensor_data.items() if isinstance(x, sensor.Image) and x.type == 'SceneFinal']

        labels = [x for name, x in sensor_data.items() if
                  isinstance(x, sensor.Image) and x.type == 'SemanticSegmentation']
        depths = [x for name, x in sensor_data.items() if
                  isinstance(x, sensor.Image) and x.type == 'Depth']

        for i in range(max(len(rgbs), len(labels), len(depths))):
            # print("Recording!!!")
            if self._current_pos_on_file == self._number_images_per_file:
                self._current_file_number += 1
                self._current_pos_on_file = 0
                pickle.dump(self.nonplayer_agents_info, self._current_np, -1)
                self._current_hf.close()
                self._current_np.close()
                self._current_hf = self._create_new_db()
                self._current_np = open(self._file_prefix + 'data_' + str(self._current_file_number).zfill(5) + '.pkl',
                                        'wb')

            pos = self._current_pos_on_file

            capture_time = int(round(time.time() * 1000))

            # print int(round(time.time() * 1000))
            # if self._record_image:
            #	im = Image.fromarray(image)
            #	b, g, r,a = im.split()
            #	im = Image.merge("RGB", (r, g, b))
            #	im.save(self._images_writing_folder_vec[folder_num] + str((capture_time)) + ".jpg")

            '''if self._record_image:
                if len(rgbs) > i:
                    im = Image.fromarray(rgbs[i])
                    b, g, r, a = im.split()
                    im = Image.merge("RGB", (r, g, b))
                    im.save(self._images_writing_folder_vec[i] + "img_" + str((capture_time)) + ".png")
                if len(labels) > i:
                    scene_seg = (labels[i][:, :, 2])
                    Image.fromarray(scene_seg * m.floor(255 / (self._number_of_seg_classes - 1))).convert('RGB').save(
                        self._images_writing_folder_vec[i] + "seg_" + str((capture_time)) + ".png")'''

            # TODO: resize

            if self._record_image_hdf5:

                # Check if there is RGB images
                if len(rgbs) > i:
                    image_RGB = to_rgb_array(rgbs[i])
                    im = Image.fromarray(image_RGB)
                    im.save("imgs/img_" + str((capture_time)) + "rgb.png")

                    image = image_RGB[self._image_cut[0]:self._image_cut[1], :, :3]
                    # image = image[:, :, :]
                    # image = scipy.misc.imresize(image, [self._image_size2, self._image_size1])
                    # im = Image.fromarray(image)
                    # im.save("imgs/img_" + str((capture_time)) + "rgbsmall.png")
                    self.data_center[pos] = image
                # Image.fromarray(image).save(self._images_writing_folder_vec[i] + "h5img_" + str((capture_time)) + ".png")

                # Check if there is semantic segmentation images
                if len(labels) > i:
                    image_SEG = labels_to_array(labels[i])
                    # im = Image.fromarray(image_SEG)
                    # im.save("imgs/img_" + str((capture_time)) + "seg.png")
                    # print(image_SEG)
                    image = []
                    im_raw = []
                    for row in range(self._image_size2):
                        image.append([])
                        im_raw.append([])
                        for column in range(self._image_size1):
                            image[row].append([image_SEG[row][column]])
                            im_raw[row].append(
                                [20 * image_SEG[row][column], image_SEG[row][column] * 20, image_SEG[row][column] * 20])
                    im = Image.fromarray(np.uint8(im_raw))
                    im.save("imgs/img_" + str((capture_time)) + "seg.png")
                    # image = image_SEG[self._image_cut[0]:self._image_cut[1], :]
                    # image = scipy.misc.imresize(image, [self._image_size2, self._image_size1], interp='nearest')
                    # image = image[:, :, np.newaxis]
                    # im = Image.fromarray(image)
                    # im.save("imgs/img_" + str((capture_time)) + "segsmall.png")
                    self.segs_center[pos] = image
                    '''
                    scene_seg = labels[i][self._image_cut[0]:self._image_cut[1], :, 2]

                    scene_seg = scipy.misc.imresize(scene_seg, [self._image_size2, self._image_size1], interp='nearest')
                    scene_seg = scene_seg[:, :, np.newaxis]
                    
                    self.segs_center[pos] = scene_seg'''
                # for layer in range(scene_seg_hot.shape[2]):
                #	Image.fromarray(scene_seg_hot[:,:,e]*255).convert('RGB').save(self._images_writing_folder_vec[i] \
                #	+ "h5seg_" + str((capture_time)) + "_" + str(layer) + ".png")
                if len(depths) > i:
                    im = Image.fromarray(to_rgb_array(depths[i]))
                    image_DEP = depth_to_array(depths[i])
                    # im = Image.fromarray(image_SEG)
                    im.save("imgs/img_" + str((capture_time)) + "dep.png")
                    image = []
                    im_raw = []
                    for row in range(self._image_size2):
                        image.append([])
                        im_raw.append([])
                        for column in range(self._image_size1):
                            d = image_DEP[row][column]
                            image[row].append([d])
                            im_raw[row].append([255 * d, 255 * d, 255 * d])
                    im = Image.fromarray(np.uint8(im_raw))
                    im.save("imgs/img_" + str((capture_time)) + "depgrey.png")
                    # image = image_[self._image_cut[0]:self._image_cut[1], :, :2]
                    # image = image[:, :, ::-1]
                    # image = scipy.misc.imresize(image, [self._image_size2, self._image_size1], interp='nearest')
                    # image = image[:, :, np.newaxis]
                    # im = Image.fromarray(image)
                    # im.save("imgs/img_" + str((capture_time)) + "rgbsmall.png")
                    self.depth_center[pos] = image
                    # depth = depths[i][self._image_cut[0]:self._image_cut[1], :, :3]

                    # depth = scipy.misc.imresize(depth, [self._image_size2, self._image_size1])
                    # self.depth_center[pos] = depth

            self.data_rewards[pos, 0] = actions.steer
            self.data_rewards[pos, 1] = actions.throttle
            self.data_rewards[pos, 2] = actions.brake
            self.data_rewards[pos, 3] = actions.hand_brake
            self.data_rewards[pos, 4] = actions.reverse
            self.data_rewards[pos, 5] = action_noise.steer
            self.data_rewards[pos, 6] = action_noise.throttle
            self.data_rewards[pos, 7] = action_noise.brake
            self.data_rewards[pos, 8] = measurements.player_measurements.transform.location.x
            self.data_rewards[pos, 9] = measurements.player_measurements.transform.location.y
            self.data_rewards[pos, 10] = measurements.player_measurements.forward_speed
            self.data_rewards[pos, 11] = measurements.player_measurements.collision_other
            self.data_rewards[pos, 12] = measurements.player_measurements.collision_pedestrians
            self.data_rewards[pos, 13] = measurements.player_measurements.collision_vehicles
            self.data_rewards[pos, 14] = measurements.player_measurements.intersection_otherlane
            self.data_rewards[pos, 15] = measurements.player_measurements.intersection_offroad
            self.data_rewards[pos, 16] = measurements.player_measurements.acceleration.x
            self.data_rewards[pos, 17] = measurements.player_measurements.acceleration.y
            self.data_rewards[pos, 18] = measurements.player_measurements.acceleration.z
            self.data_rewards[pos, 19] = measurements.platform_timestamp
            self.data_rewards[pos, 20] = measurements.game_timestamp
            self.data_rewards[pos, 21] = measurements.player_measurements.transform.orientation.x
            self.data_rewards[pos, 22] = measurements.player_measurements.transform.orientation.y
            self.data_rewards[pos, 23] = measurements.player_measurements.transform.orientation.z
            self.data_rewards[pos, 24] = direction
            self.data_rewards[pos, 25] = i
            self.data_rewards[pos, 26] = float(self._camera_dict[i][1])
            self.data_rewards[pos, 27] = measurements.player_measurements.transform.rotation.pitch
            self.data_rewards[pos, 28] = measurements.player_measurements.transform.rotation.roll
            self.data_rewards[pos, 29] = measurements.player_measurements.transform.rotation.yaw
            if self._record_waypoints:
                self.data_rewards[pos, 30] = waypoints[0][0]
                self.data_rewards[pos, 31] = waypoints[0][1]
                self.data_rewards[pos, 32] = waypoints[1][0]
                self.data_rewards[pos, 33] = waypoints[1][1]

            # print 'Angle ',self.data_rewards[pos,26]
            # print 'LENS ',len(images.rgb),len(images.scene_seg)
            self.nonplayer_agents_info.append({})
            for agent in measurements.non_player_agents:
                if agent.HasField('vehicle'):
                    if not self.nonplayer_agents_info[pos].has_key('vehicle'):
                        self.nonplayer_agents_info[pos]['vehicle'] = []
                    self.nonplayer_agents_info[pos]['vehicle'].append(
                        [agent.id, agent.vehicle.forward_speed, agent.vehicle.transform.location.x,
                         agent.vehicle.transform.location.y, agent.vehicle.transform.location.z,
                         agent.vehicle.transform.orientation.x, agent.vehicle.transform.orientation.y,
                         agent.vehicle.transform.orientation.z, agent.vehicle.transform.rotation.pitch,
                         agent.vehicle.transform.rotation.roll, agent.vehicle.transform.rotation.yaw ])
                elif agent.HasField('pedestrian'):
                    if not self.nonplayer_agents_info[pos].has_key('pedestrian'):
                        self.nonplayer_agents_info[pos]['pedestrian'] = []
                    self.nonplayer_agents_info[pos]['pedestrian'].append(
                        [agent.id, agent.pedestrian.forward_speed, agent.pedestrian.transform.location.x,
                         agent.pedestrian.transform.location.y, agent.pedestrian.transform.location.z,
                         agent.pedestrian.transform.orientation.x, agent.pedestrian.transform.orientation.y,
                         agent.pedestrian.transform.orientation.z, agent.pedestrian.transform.rotation.pitch,
                         agent.pedestrian.transform.rotation.roll, agent.pedestrian.transform.rotation.yaw])
                    #print agent.pedestrian.transform.rotation.pitch, agent.pedestrian.transform.rotation.roll, agent.pedestrian.transform.rotation.yaw
                elif agent.HasField('traffic_light'):
                    if not self.nonplayer_agents_info[pos].has_key('traffic_light'):
                        self.nonplayer_agents_info[pos]['traffic_light'] = []
                    self.nonplayer_agents_info[pos]['traffic_light'].append(
                        [agent.id, agent.traffic_light.state, agent.traffic_light.transform.location.x,
                         agent.traffic_light.transform.location.y, agent.traffic_light.transform.location.z])
                elif agent.HasField('speed_limit_sign'):
                    if not self.nonplayer_agents_info[pos].has_key('speed_limit_sign'):
                        self.nonplayer_agents_info[pos]['speed_limit_sign'] = []
                    self.nonplayer_agents_info[pos]['speed_limit_sign'].append(
                        [agent.id, agent.speed_limit_sign.speed_limit, agent.speed_limit_sign.transform.location.x,
                         agent.speed_limit_sign.transform.location.y, agent.speed_limit_sign.transform.location.z])

            self._current_pos_on_file += 1

        # fo.close()

    def close(self):

        self._current_hf.close()
