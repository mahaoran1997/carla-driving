import os
from pid_controller.pid import PID
import random


# TODO: Turn this into an YML configuration file anyway.

class ConfigAutopilot:

    # GENERAL THINGS

    def __init__(self, city_name):
        # WAYPOINT ----- TRAJECTORY CONFIGURATION

        dir_path = os.path.dirname(__file__)
        self.city_file = dir_path + '/../planner/' + city_name + '.txt'
        self.city_map_file = dir_path + '/../planner/' + city_name + '.png'
        self.city_name = city_name


        extra_spacing = (random.randint(0, 4) - 2)
        self.lane_shift_distance = 13 + extra_spacing  # The amount of shifting from the center the car should go
        self.extra_spacing_rights = -2 - extra_spacing
        self.extra_spacing_lefts = 10 + extra_spacing
        self.way_key_points_predicted = 7
        self.number_of_waypoints = 30

        # CONTROLLER ----- PARAMETER CONFIGURATION

        # Default Throttle
        self.throttle = 0.5
        # Default Brake
        self.brake = 0
        # Gain on computed steering angle
        self.steer_gain = 0.7
        # Distance Threshold Traffic Light
        self.tl_dist_thres = 1500
        # Angle Threshold Traffic Light
        self.tl_angle_thres = 0.5
        # Distance Threshold Pedestrian
        self.p_dist_thres = 1000
        # Angle Threshold Pedestrian
        self.p_angle_thres = 0.3
        # Distance Threshold Vehicle
        self.v_dist_thres = 1500
        # Angle Threshold Vehicle
        self.v_angle_thres = 0.35
        # Select WP Number
        self.wp_num_steer = 0.8  # Select WP - Reverse Order: 1 - closest, 0 - furthest
        self.wp_num_speed = 0.5  # Select WP - Reverse Order: 1 - closest, 0 - furthest

        # Stop for traffic lights
        self.stop4TL = False
        # Stop for pedestrians
        self.stop4P = True
        # Stop for vehicles
        self.stop4V = True

        # Strength for applying brake - Value between 0 and 1
        self.brake_strength = 1
        # Factor to control coasting
        self.coast_factor = 2
        # PID speed controller
        self.pid = PID(p=0.25, i=0.08, d=0)
        # Target speed - could be controlled by speed limit
        self.target_speed = 35
        # Maximum throttle
        self.throttle_max = 0.75
        # Flag to decide if PID should be used
        self.usePID = True  # If off car will be a bit more aggressive
