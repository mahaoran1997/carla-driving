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
                 record_image=True, image_cut=[0, 600], camera_dict={}, record_waypoints=False):

        self._number_of_seg_classes = 13
        self._record_image_hdf5 = True
        self._number_images_per_file = 200
        self._file_prefix = file_prefix
        self._image_size2 = resolution[1]
        self._image_size1 = resolution[0]
        self._record_image = False
        self._number_rewards = 27 + 4 * record_waypoints
        self._image_cut = image_cut
        self._camera_dict = camera_dict
        self._record_waypoints = record_waypoints
        if not os.path.exists(self._file_prefix):
            os.mkdir(self._file_prefix)

        self._current_file_number = current_file_number
        self.data = []
        #self._current_hf = self._create_new_db()

        #self._current_np = open(self._file_prefix + 'data_' + str(self._current_file_number).zfill(5) + '.pkl', 'wb')
        self._images_writing_folder = self._file_prefix + 'imgs/'
        #if not os.path.exists(self._images_writing_folder):
        #    os.mkdir(self._images_writing_folder)
        # self._images_writing_folder_vec = [self._images_writing_folder + 'rgbs/', self._images_writing_folder + 'labels/', self._images_writing_folder + 'depths/']
        # for i in range

        self._images_writing_folder_vec = []

        if self._record_image:
            for i in range(max(len(self._camera_dict)/3, 1)):
                images_writing_folder = self._file_prefix + "Images_" + str(i) + "/"
                if not os.path.exists(images_writing_folder):
                    os.mkdir(images_writing_folder)
                self._images_writing_folder_vec.append(images_writing_folder)

        #self._csv_writing_file = self._file_prefix + 'outputs.csv'
        self._current_pos_on_file = 0
        self._data_queue = Queue(5000)
        self.run_disk_writer()

    def record(self, measurements, action, action_noise, direction, waypoints=None, new = False, recording = True):
        self._write_to_disk([measurements, action, action_noise, direction, waypoints, new, recording])
        #self._data_queue.put([measurements, action, action_noise, direction, waypoints, new])

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

        while True:
            data = self._data_queue.get()
            if self._data_queue.qsize() % 100 == 0:
                print "QSIZE:",self._data_queue.qsize()
            self._write_to_disk(data)

    def _write_to_disk(self, data):
        # Use the dictionary for this

        measurements = data[0]
        actions = data[1]
        action_noise = data[2]
        direction = data[3]
        waypoints = data[4]
        new = data[5]
        recording = data[6]

        print('new')
        print(new)
        print(self._current_file_number)

        if new:
            if self._current_file_number != 0:
                #print 
                current_np = open(self._file_prefix + 'data_' + str(self._current_file_number).zfill(5) + '.pkl', 'wb')
                pickle.dump(self.data, current_np, True)
                current_np.close()
            
            self._current_file_number += 1

            self.data = []



        if not recording:
            return

        current_data = {}

        capture_time = int(round(time.time() * 1000))

        i = 0

        # print int(round(time.time() * 1000))
        # if self._record_image:
        #	im = Image.fromarray(image)
        #	b, g, r,a = im.split()
        #	im = Image.merge("RGB", (r, g, b))
        #	im.save(self._images_writing_folder_vec[folder_num] + str((capture_time)) + ".jpg")
        if self._record_image:
            if len(measurements['BGRA']) > 0:
                im = Image.fromarray(measurements['BGRA'][i])
                b, g, r, a = im.split()
                im = Image.merge("RGB", (r, g, b))
                im.save(self._images_writing_folder_vec[i] + "img_" + str((capture_time)) + ".png")
            if len(measurements['Labels']) > 0:
                scene_seg = (measurements['Labels'][i][:, :, 2])
                Image.fromarray(scene_seg * m.floor(255 / (self._number_of_seg_classes - 1))).convert('RGB').save(
                    self._images_writing_folder_vec[i] + "seg_" + str((capture_time)) + ".png")
            if len(measurements['Depth']) > 0:
                im = Image.fromarray(measurements['Depth'][i])
                b, g, r, a = im.split()
                im = Image.merge("RGB", (r, g, b))
                im.save(self._images_writing_folder_vec[i] + "dep_" + str((capture_time)) + ".png")



        
        if self._record_image_hdf5:

            # Check if there is RGB images
            if len(measurements['BGRA']) > i:
                image = measurements['BGRA'][i][self._image_cut[0]:self._image_cut[1], :, :3]
                image = image[:, :, ::-1]
                image = scipy.misc.imresize(image, [self._image_size2, self._image_size1])
                current_data['rgb'] = image
            # Image.fromarray(image).save(self._images_writing_folder_vec[i] + "h5img_" + str((capture_time)) + ".png")

            # Check if there is semantic segmentation images
            if len(measurements['Labels']) > i:
                scene_seg = measurements['Labels'][i][self._image_cut[0]:self._image_cut[1], :, 2]

                scene_seg = scipy.misc.imresize(scene_seg, [self._image_size2, self._image_size1], interp='nearest')
                scene_seg = scene_seg[:, :, np.newaxis]

                current_data['seg'] = image
            # for layer in range(scene_seg_hot.shape[2]):
            #	Image.fromarray(scene_seg_hot[:,:,e]*255).convert('RGB').save(self._images_writing_folder_vec[i] \
            #	+ "h5seg_" + str((capture_time)) + "_" + str(layer) + ".png")
            if len(measurements['Depth']) > i:
                depth = measurements['Depth'][i][self._image_cut[0]:self._image_cut[1], :, :3]

                depth = scipy.misc.imresize(depth, [self._image_size2, self._image_size1])
                current_data['dep'] = image
        

        data_rewards = [None] * 28

        data_rewards[0] = actions.steer
        data_rewards[1] = actions.throttle
        data_rewards[2] = actions.brake
        data_rewards[3] = actions.hand_brake #
        data_rewards[4] = actions.reverse
        data_rewards[5] = action_noise.steer
        data_rewards[6] = action_noise.throttle
        data_rewards[7] = action_noise.brake
        data_rewards[8] = measurements['PlayerMeasurements'].transform.location.x
        data_rewards[9] = measurements['PlayerMeasurements'].transform.location.y
        data_rewards[10] = measurements['PlayerMeasurements'].transform.location.z
        data_rewards[11] = measurements['PlayerMeasurements'].forward_speed
        data_rewards[12] = measurements['PlayerMeasurements'].collision_other
        data_rewards[13] = measurements['PlayerMeasurements'].collision_pedestrians
        data_rewards[14] = measurements['PlayerMeasurements'].collision_vehicles
        data_rewards[15] = measurements['PlayerMeasurements'].intersection_otherlane
        data_rewards[16] = measurements['PlayerMeasurements'].intersection_offroad
        data_rewards[17] = measurements['PlayerMeasurements'].acceleration.x
        data_rewards[18] = measurements['PlayerMeasurements'].acceleration.y
        data_rewards[19] = measurements['PlayerMeasurements'].acceleration.z
        data_rewards[20] = measurements['WallTime']
        data_rewards[21] = measurements['GameTime']
        data_rewards[22] = measurements['PlayerMeasurements'].transform.orientation.x
        data_rewards[23] = measurements['PlayerMeasurements'].transform.orientation.y
        data_rewards[24] = measurements['PlayerMeasurements'].transform.orientation.z
        data_rewards[25] = direction
        data_rewards[26] = i
        data_rewards[27] = float(self._camera_dict[i][1])
        if self._record_waypoints:
            data_rewards.append(waypoints[0][0])
            data_rewards.append(waypoints[0][1])
            data_rewards.append(waypoints[1][0])
            data_rewards.append(waypoints[1][1])

        #self.nonplayer_agents_info.append({})
        nonplayer_agents_info = {}
        for agent in measurements['Agents']:
            if agent.HasField('vehicle'):
                #if (agent.vehicle.transform.location.x == measurements['PlayerMeasurements'].transform.location.x and agent.vehicle.transform.location.y)
                if not nonplayer_agents_info.has_key('vehicle'):
                    nonplayer_agents_info['vehicle'] = []
                nonplayer_agents_info['vehicle'].append(
                    [agent.id, agent.vehicle.forward_speed, agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y, agent.vehicle.transform.location.z,
                        agent.vehicle.transform.orientation.x, agent.vehicle.transform.orientation.y,
                        agent.vehicle.bounding_box.transform.location.x,
                        agent.vehicle.bounding_box.transform.location.y,
                        agent.vehicle.bounding_box.transform.location.z,
                        agent.vehicle.bounding_box.transform.orientation.x, 
                        agent.vehicle.bounding_box.transform.orientation.y, 
                        agent.vehicle.bounding_box.transform.orientation.z, 
                        agent.vehicle.bounding_box.extent.x,
                        agent.vehicle.bounding_box.extent.y,
                        agent.vehicle.bounding_box.extent.z
                        ])
            elif agent.HasField('pedestrian'):
                if not nonplayer_agents_info.has_key('pedestrian'):
                    nonplayer_agents_info['pedestrian'] = []
                nonplayer_agents_info['pedestrian'].append(
                    [agent.id, agent.pedestrian.forward_speed, agent.pedestrian.transform.location.x,
                        agent.pedestrian.transform.location.y, agent.pedestrian.transform.location.z,
                        agent.pedestrian.transform.orientation.x, agent.pedestrian.transform.orientation.y,
                        agent.pedestrian.bounding_box.transform.location.x,
                        agent.pedestrian.bounding_box.transform.location.y,
                        agent.pedestrian.bounding_box.transform.location.z,
                        agent.pedestrian.bounding_box.transform.orientation.x, 
                        agent.pedestrian.bounding_box.transform.orientation.y, 
                        agent.pedestrian.bounding_box.transform.orientation.z, 
                        agent.pedestrian.bounding_box.extent.x,
                        agent.pedestrian.bounding_box.extent.y,
                        agent.pedestrian.bounding_box.extent.z
                        ])
                # print agent.pedestrian.transform.rotation.pitch, agent.pedestrian.transform.rotation.roll, agent.pedestrian.transform.rotation.yaw
            elif agent.HasField('traffic_light'):
                if not nonplayer_agents_info.has_key('traffic_light'):
                    nonplayer_agents_info['traffic_light'] = []
                nonplayer_agents_info['traffic_light'].append(
                    [agent.id, agent.traffic_light.state, agent.traffic_light.transform.location.x,
                        agent.traffic_light.transform.location.y, agent.traffic_light.transform.location.z])
            elif agent.HasField('speed_limit_sign'):
                if not nonplayer_agents_info.has_key('speed_limit_sign'):
                    nonplayer_agents_info['speed_limit_sign'] = []
                nonplayer_agents_info['speed_limit_sign'].append(
                    [agent.id, agent.speed_limit_sign.speed_limit, agent.speed_limit_sign.transform.location.x,
                        agent.speed_limit_sign.transform.location.y, agent.speed_limit_sign.transform.location.z])
        current_data['data_rewards'] = data_rewards
        current_data['nonplayer'] = nonplayer_agents_info
        self.data.append(current_data)

    def close(self):

        current_np = open(self._file_prefix + 'data_' + str(self._current_file_number).zfill(5) + '.pkl', 'wb')
        pickle.dump(self.data, current_np, True)
        current_np.close()
