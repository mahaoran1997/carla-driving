import h5py, glob, os, math, sys
import numpy as np

sys.path.append('drive_interfaces/carla/carla_client')
from carla.planner.planner import Planner

input_id = "noiser_3cam"
output_id = "noiser_3cam_direction"
debug_start = 0
debug_end= 14000000
use_3_cam = False
copy_3cam = True

all_files = glob.glob("/data/yang/code/aws/scratch/carla_collect/"+str(input_id)+"/*/data_*.h5")

# first read all pos and ori
pos = [] # or location
ori = []
for one_h5 in sorted(all_files)[debug_start:debug_end]:
    print(one_h5)
    hin = h5py.File(one_h5, 'r')
    pos.append(hin["targets"][:, 8: 10])
    ori.append(hin["targets"][:, 21:24])

pos0 = np.concatenate(pos, axis=0)
pos = np.zeros((pos0.shape[0], 3))
pos[:, 0:2] = pos0
pos[:, 2] = 0.22
ori = np.concatenate(ori, axis=0)

# instantiate a planner
CityName = "Town01"
planner = Planner(CityName)

# compute is this position away from an intersection?
is_away_from_inter = []
for i in range(pos.shape[0]):
    res = planner.test_position(pos[i, :])
    is_away_from_inter.append(res)
is_away_from_inter = np.array(is_away_from_inter)


sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def get_command(index, planner_in, last_end):
    if index == pos.shape[0]-1:
        # this is the last one, return follow
        return 2, None, -1

    if index > 0:
        # when we have a new trajectory
        if sldist(pos[index - 1], pos[index]) > 0.2 * 9.7 * 1.5:
            planner_in = Planner(CityName)
            print("starting a new trajectory")

    last_inter = is_away_from_inter[index]
    count_inter_changes = 0

    #end_index = last_end - 1
    end_index = index
    while True:
        end_index += 1

        if sldist(pos[end_index - 1], pos[end_index]) > 0.2 * 9.7 * 1.5 or \
            end_index == pos.shape[0] - 1:

            if sldist(pos[end_index - 1], pos[end_index]) > 0.2 * 9.7 * 1.5:
                end_index -= 1

            direction = planner.get_next_command(pos[index],
                                                 ori[index],
                                                 pos[end_index],
                                                 ori[end_index])

            if math.fabs(direction)<0.1:
                direction = 2.0

            return direction, planner_in, end_index

        if last_inter != is_away_from_inter[end_index]:
            last_inter = not last_inter
            count_inter_changes += 1

        if count_inter_changes < 3:
            continue

        direction = planner.get_next_command(pos[index],
                                             ori[index],
                                             pos[end_index],
                                             ori[end_index])

        if math.fabs(direction) > 0.1:
            return direction, planner_in, end_index

planner_in = Planner(CityName)
last_end = 1 # TODO: not finished

counter = 0
for one_h5 in sorted(all_files)[debug_start:debug_end]:
    target_path = one_h5.replace("/"+str(input_id)+"/", "/"+str(output_id)+"/")
    print("converting ", one_h5, " to ", target_path)
    dirname = os.path.dirname(target_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    hf = h5py.File(target_path, 'w')
    factor = 3*use_3_cam + 1*(not use_3_cam)
    data_rewards = hf.create_dataset('targets', (200*factor, 35), 'f')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    sensor = hf.create_dataset("CameraMiddle", (200*factor,), dtype=dt)
    if copy_3cam:
        sensorL = hf.create_dataset("CameraLeft", (200 * factor,), dtype=dt)
        sensorR = hf.create_dataset("CameraRight", (200 * factor,), dtype=dt)

    hin = h5py.File(one_h5, 'r')
    count_within_file = 0
    for i in range(200):
        target_line = hin["targets"][i, :]
        direction, planner_in, last_end = get_command(counter, planner_in, last_end)
        target_line[24] = direction
        counter += 1

        # middle
        data_rewards[count_within_file, :] = target_line
        sensor[count_within_file] = hin["CameraMiddle"][i]
        if copy_3cam:
            sensorL[count_within_file] = hin["CameraLeft"][i]
            sensorR[count_within_file] = hin["CameraRight"][i]
        count_within_file += 1

        if use_3_cam:
            speed_pos = 10
            steer_pos = 0

            angle = math.radians(30.0)

            time_use = 1.0
            car_lenght = 6.0
            speed = math.fabs(hin["targets"][i, speed_pos]) * 3.6
            delta = min(6 * (math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / math.pi, 0.3)
            # TODO: this is an empirical good number
            delta = 0.42

            # left
            data_rewards[count_within_file, :] = target_line
            data_rewards[count_within_file, steer_pos] += delta
            sensor[count_within_file] = hin["CameraLeft"][i]
            count_within_file += 1

            # right
            data_rewards[count_within_file, :] = target_line
            data_rewards[count_within_file, steer_pos] -= delta
            sensor[count_within_file] = hin["CameraRight"][i]
            count_within_file += 1




    hin.close()
    hf.close()
