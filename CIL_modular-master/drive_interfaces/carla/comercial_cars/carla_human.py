import pygame, io, math, time
from configparser import ConfigParser
import numpy as np

from carla.planner.planner import Planner
from carla.client import VehicleControl
#from carla.client import make_carla_client
from carla.client import CarlaClient

from driver import Driver

sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def v3d_to_array(v3d):
    return [v3d.x, v3d.y, v3d.z]

def find_valid_episode_position(positions, planner):
    while True:
        print("a new trail to find a start and end position")
        index_start = np.random.randint(len(positions))
        start_pos = positions[index_start]
        # The test position func means is_away_from_intersection
        if not planner.test_position(v3d_to_array(start_pos.location)):
            continue

        index_goal = np.random.randint(len(positions))
        goals_pos = positions[index_goal]
        print((' TESTING start and end location (', index_start, ',', index_goal, ')'))
        if not planner.test_position(v3d_to_array(goals_pos.location)):
            continue

        dist = sldist([start_pos.location.x, start_pos.location.y], [goals_pos.location.x, goals_pos.location.y])
        if dist < 250.0:
            print("continue because distance too close", dist, start_pos, goals_pos)
            continue

        if planner.is_there_posible_route(v3d_to_array(start_pos.location), v3d_to_array(start_pos.orientation),
                                          v3d_to_array(goals_pos.location), v3d_to_array(goals_pos.orientation)):
            break

    return index_start, index_goal


class CarlaHuman(Driver):
    def __init__(self, driver_conf):
        Driver.__init__(self)
        # some behaviors
        self._autopilot = driver_conf.autopilot
        self._reset_period = driver_conf.reset_period # those reset period are in the actual system time, not in simulation time
        self._goal_reaching_threshold = 3
        self.use_planner = driver_conf.use_planner
        # we need the planner to find a valid episode, so we initialize one no matter what
        self.planner = Planner(driver_conf.city_name)

        # resources
        self._host = driver_conf.host
        self._port = driver_conf.port

        # various config files
        self._driver_conf = driver_conf
        self._config_path = driver_conf.carla_config

        # some initializations
        self._straight_button = False
        self._left_button = False
        self._right_button = False

        self._rear = False
        self._recording = False
        self._skiped_frames = 20
        self._stucked_counter = 0

    def start(self):
        self.carla = CarlaClient(self._host, int(self._port), timeout=120)
        self.carla.connect()

        self._reset()

        if not self._autopilot:
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                print("Please Connect Just One Joystick")
                raise

            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def __del__(self):
        if hasattr(self, 'carla'):
            print("destructing the connection")
            self.carla.disconnect()

    def _reset(self):
        self._start_time = time.time()

        # create the carla config based on template and the params passed in
        config = ConfigParser()
        config.optionxform = str
        config.read(self._config_path)
        config.set('CARLA/LevelSettings', 'NumberOfVehicles',    self._driver_conf.cars)
        config.set('CARLA/LevelSettings', 'NumberOfPedestrians', self._driver_conf.pedestrians)
        config.set('CARLA/LevelSettings', 'WeatherId',           self._driver_conf.weather)
        output = io.StringIO()
        config.write(output)
        scene_descriptions = self.carla.load_settings(output.getvalue())

        # based on the scene descriptions, find the start and end positions
        self.positions = scene_descriptions.player_start_spots
        # the episode_config saves [start_index, end_index]
        self.episode_config = find_valid_episode_position(self.positions, self.planner)

        self.carla.start_episode(self.episode_config[0])
        print('RESET ON POSITION ', self.episode_config[0], ", the target location is: ", self.episode_config[1])
        self._skiped_frames = 0
        self._stucked_counter = 0

    def _get_direction_buttons(self):
        # with suppress_stdout():
        if (self.joystick.get_button(6)):
            self._left_button = False
            self._right_button = False
            self._straight_button = False

        if (self.joystick.get_button(5)):
            self._left_button = True
            self._right_button = False
            self._straight_button = False

        if (self.joystick.get_button(4)):
            self._right_button = True
            self._left_button = False
            self._straight_button = False

        if (self.joystick.get_button(7)):
            self._straight_button = True
            self._left_button = False
            self._right_button = False

        return [self._left_button, self._right_button, self._straight_button]

    def get_recording(self):
        if self._autopilot:
            if self._skiped_frames >= 20:
                return True
            else:
                self._skiped_frames += 1
                return False

        else:
            if (self.joystick.get_button(8)):
                self._recording = True
            if (self.joystick.get_button(9)):
                self._recording = False

            return self._recording

    def get_reset(self):
        if self._autopilot:
            # increase the stuck detector if conditions satisfy
            if self._latest_measurements.player_measurements.forward_speed < 0.1:
                self._stucked_counter += 1
            else:
                self._stucked_counter = 0

            # if within auto pilot, reset if long enough or has collisions
            if time.time() - self._start_time > self._reset_period \
              or self._latest_measurements.player_measurements.collision_vehicles    > 0.0 \
              or self._latest_measurements.player_measurements.collision_pedestrians > 0.0 \
              or self._latest_measurements.player_measurements.collision_other       > 0.0 \
              or self._stucked_counter > 150:
                if self._stucked_counter > 150:
                    reset_because_stuck = True
                else:
                    reset_because_stuck = False

                self._reset()

                if reset_because_stuck:
                    print("resetting because getting stucked.....")
                    return True
        else:
            if (self.joystick.get_button(4)):
                self._reset()

        return False

    def get_waypoints(self):
        # TODO: waiting for German Ros to expose the waypoints
        wp1 = [1.0, 1.0]
        wp2 = [2.0, 2.0]
        return [wp1, wp2]

    def compute_action(self, sensor, speed):
        if not self._autopilot:
            steering_axis = self.joystick.get_axis(0)
            acc_axis = self.joystick.get_axis(2)
            brake_axis = self.joystick.get_axis(3)

            if (self.joystick.get_button(3)):
                self._rear = True
            if (self.joystick.get_button(2)):
                self._rear = False

            control = VehicleControl()
            control.steer = steering_axis
            control.throttle = -(acc_axis - 1) / 2.0
            control.brake = -(brake_axis - 1) / 2.0
            if control.brake < 0.001:
                control.brake = 0.0
            control.hand_brake = 0
            control.reverse = self._rear
        else:
            # This relies on the calling of get_sensor_data, otherwise self._latest_measurements are not filled
            control = self._latest_measurements.player_measurements.autopilot_control

        return control

    def get_sensor_data(self, goal_pos=None, goal_ori=None):
        # return the latest measurement and the next direction
        measurements, sensor_data = self.carla.read_data()
        self._latest_measurements = measurements

        if self.use_planner:
            player_data = measurements.player_measurements
            pos = [player_data.transform.location.x,
                   player_data.transform.location.y,
                   0.22]
            ori = [player_data.transform.orientation.x,
                   player_data.transform.orientation.y,
                   player_data.transform.orientation.z]

            if sldist([player_data.transform.location.x,
                       player_data.transform.location.y],
                      [self.positions[self.episode_config[1]].location.x,
                       self.positions[self.episode_config[1]].location.y]) < self._goal_reaching_threshold:
                self._reset()

            direction = self.planner.get_next_command(pos, ori,
                                                      [self.positions[self.episode_config[1]].location.x,
                                                       self.positions[self.episode_config[1]].location.y,
                                                       0.22],
                                                      (1, 0, 0))
            print("planner output direction: ", direction)
        else:
            direction = 2.0

        return measurements, sensor_data, direction

    def act(self, control):
        self.carla.send_control(control)
