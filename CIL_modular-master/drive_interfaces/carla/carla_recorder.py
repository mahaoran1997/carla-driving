import os, h5py, scipy, cv2, math, sys, time
import numpy as np
from threading import Thread
from queue import Queue
sys.path.append('drive_interfaces/carla/carla_client')
from carla import image_converter

# lets put a big queue for the disk. So I keep it real time while the disk is writing stuff
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread

    return wrapper


def get_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec / dist, dist


def get_angle(vec_dst, vec_src):
    angle = math.atan2(vec_dst[1], vec_dst[0]) - math.atan2(vec_src[1], vec_src[0])
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


class Recorder(object):
    # We assume a three camera case not many cameras per input ....
    def __init__(self, file_prefix, resolution=[800, 600], current_file_number=0,
                image_cut=[0, 600]):
        self._file_prefix = file_prefix
        if not os.path.exists(self._file_prefix):
            os.mkdir(self._file_prefix)

        # image related storing options
        self._number_images_per_file = 200
        self._image_size1 = resolution[0]
        self._image_size2 = resolution[1]
        self._image_cut = image_cut
        #self._sensor_names = ['CameraLeft', 'CameraMiddle', 'CameraRight', 'SegLeft', 'SegMiddle', 'SegRight', 'DepthLeft', 'DepthMiddle', 'DepthRight']
        self._sensor_names = ['CameraLeft', 'CameraMiddle', 'CameraRight']

        # other rewards
        self._number_rewards = 35

        # initialize for writing the db
        self._current_file_number = current_file_number
        self._current_pos_on_file = 0
        self._current_hf = self._create_new_db()
        self._data_queue = Queue(5000)
        self.run_disk_writer()
        self._finish_writing = True

    def hf_path_formatter(self, id):
        path = self._file_prefix + 'data_' + str(id).zfill(5) + '.h5'
        return path

    def _create_new_db(self):
        # TODO: change the reading part to have the image decoding
        self._current_hf_path = self.hf_path_formatter(self._current_file_number)
        hf = h5py.File(self._current_hf_path, 'w')
        self.data_rewards = hf.create_dataset('targets', (self._number_images_per_file, self._number_rewards), 'f')
        self.sensors={}
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        for sensor_name in self._sensor_names:
            self.sensors[sensor_name] = hf.create_dataset(sensor_name, (self._number_images_per_file,), dtype=dt)

        return hf

    def record(self, measurements, sensor_data, action, action_noise, direction, waypoints=None):
        self._data_queue.put([measurements, sensor_data, action, action_noise, direction, waypoints])

    @threaded
    def run_disk_writer(self):
        while True:
            data = self._data_queue.get()
            # if self._data_queue.qsize() % 100 == 0:
            # print "QSIZE:",self._data_queue.qsize()
            self._finish_writing = False
            self._write_to_disk(data)
            self._finish_writing = True

    def _write_to_disk(self, data):
        # Use the dictionary for this
        measurements, sensor_data, actions, action_noise, direction, waypoints = data

        if self._current_pos_on_file == self._number_images_per_file:
            self._current_file_number += 1
            self._current_pos_on_file = 0
            self._current_hf.close()
            self._current_hf = self._create_new_db()
        pos = self._current_pos_on_file

        for sensor_name in self._sensor_names:
            if "depth" in sensor_name.lower():
                image = image_converter.to_bgra_array(sensor_data[sensor_name])
                image = image[self._image_cut[0]:self._image_cut[1], :, :3]
                image = scipy.misc.imresize(image, [self._image_size2, self._image_size1])
                encoded = np.fromstring(cv2.imencode(".png", image)[1], dtype=np.uint8)
            elif "camera" in sensor_name.lower():
                image = image_converter.to_bgra_array(sensor_data[sensor_name])
                image = image[self._image_cut[0]:self._image_cut[1], :, :3]
                image = scipy.misc.imresize(image, [self._image_size2, self._image_size1])
                encoded = np.fromstring(cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1], dtype=np.uint8)
            elif "seg" in sensor_name.lower():
                image = image_converter.labels_to_array(sensor_data[sensor_name])
                image = image[self._image_cut[0]:self._image_cut[1], :]
                image = scipy.misc.imresize(image, [self._image_size2, self._image_size1], interp='nearest')
                encoded = np.fromstring(cv2.imencode(".png", image)[1], dtype=np.uint8)
            else:
                raise

            self.sensors[sensor_name][pos] = encoded

        self.data_rewards[pos, 0] = actions.steer
        self.data_rewards[pos, 1] = actions.throttle
        self.data_rewards[pos, 2] = actions.brake
        self.data_rewards[pos, 3] = actions.hand_brake
        self.data_rewards[pos, 4] = actions.reverse
        self.data_rewards[pos, 5] = action_noise.steer
        self.data_rewards[pos, 6] = action_noise.throttle
        self.data_rewards[pos, 7] = action_noise.brake
        self.data_rewards[pos, 8] = measurements.player_measurements.transform.location.x # cm -> m, but this is not used anywhere
        self.data_rewards[pos, 9] = measurements.player_measurements.transform.location.y
        self.data_rewards[pos, 10] = measurements.player_measurements.forward_speed # TODO: km/h -> m/s
        self.data_rewards[pos, 11] = measurements.player_measurements.collision_other
        self.data_rewards[pos, 12] = measurements.player_measurements.collision_pedestrians
        self.data_rewards[pos, 13] = measurements.player_measurements.collision_vehicles
        self.data_rewards[pos, 14] = measurements.player_measurements.intersection_otherlane
        self.data_rewards[pos, 15] = measurements.player_measurements.intersection_offroad
        self.data_rewards[pos, 16] = measurements.player_measurements.acceleration.x # This is not used anywhere
        self.data_rewards[pos, 17] = measurements.player_measurements.acceleration.y
        self.data_rewards[pos, 18] = measurements.player_measurements.acceleration.z
        self.data_rewards[pos, 19] = measurements.platform_timestamp
        self.data_rewards[pos, 20] = measurements.game_timestamp
        self.data_rewards[pos, 21] = measurements.player_measurements.transform.orientation.x # those are deprecated, but they are not used anywhere
        self.data_rewards[pos, 22] = measurements.player_measurements.transform.orientation.y
        self.data_rewards[pos, 23] = measurements.player_measurements.transform.orientation.z
        self.data_rewards[pos, 24] = direction
        self.data_rewards[pos, 25] = 0 # originally i, now, not used, they are also not used anywhere else
        self.data_rewards[pos, 26] = 0 # originally this camera's yaw, but now not used

        # TODO: below is waypoints
        self.data_rewards[pos, 27] = waypoints[0][0]
        self.data_rewards[pos, 28] = waypoints[0][1]
        self.data_rewards[pos, 29] = waypoints[1][0]
        self.data_rewards[pos, 30] = waypoints[1][1]

        # merge the convertwp functionality into this file
        def get_angle_mag(wp0, wp1):
            wp_vector, wp_mag = get_vec_dist(wp0, wp1,
                                               measurements.player_measurements.transform.location.x,
                                               measurements.player_measurements.transform.location.y)
            if wp_mag > 0:
                # TODO: check the definition of x and y, as well as the def of camera angle
                wp_angle = get_angle(wp_vector, [measurements.player_measurements.transform.orientation.x,
                                                 measurements.player_measurements.transform.orientation.y]) - \
                            math.radians(self.data_rewards[pos, 26])
            else:
                wp_angle = 0
            return wp_angle, wp_mag

        wp1_angle, wp1_mag = get_angle_mag(waypoints[0][0], waypoints[0][1])
        wp2_angle, wp2_mag = get_angle_mag(waypoints[1][0], waypoints[1][1])

        self.data_rewards[pos, 31] = wp1_angle
        self.data_rewards[pos, 32] = wp1_mag
        self.data_rewards[pos, 33] = wp2_angle
        self.data_rewards[pos, 34] = wp2_mag

        self._current_pos_on_file += 1

    def close(self):
        while not self._data_queue.empty() or not self._finish_writing:
            print("waiting to write out data")
            time.sleep(1)
        self._current_hf.close()
        if self._current_pos_on_file != self._number_images_per_file:
            # we have an incomplete file
            print("remove the not complete file")
            os.remove(self._current_hf_path)

    def remove_current_and_previous(self):
        # remove the current one if it's not finished
        self.close()

        # in the case of the current file has finished writing, remove it as well
        if os.path.exists(self._current_hf_path):
            os.remove(self._current_hf_path)

        num_dropped = self._current_pos_on_file

        # to be safe, remove the previous file as well
        previous_path = self.hf_path_formatter(self._current_file_number - 1)
        if os.path.exists(previous_path):
            print("removing the previous database", previous_path)
            os.remove(previous_path)

            self._current_file_number -= 1
            num_dropped += self._number_images_per_file
        else:
            if self._current_file_number !=0:
                print("Warning!! there should be a previous file to remove, but I could not find it")
            else:
                print("not removing previous file because this is the 0-th file")

        self._current_pos_on_file = 0
        self._current_hf = self._create_new_db()

        return num_dropped