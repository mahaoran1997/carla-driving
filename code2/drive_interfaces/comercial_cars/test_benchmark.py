# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function

import os
import datetime
import random
import numpy as np

from carla.benchmarks.benchmark import Benchmark
from carla.benchmarks.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.planner.planner import Planner

from carla.benchmarks.metrics import compute_summary

sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

"""
def find_valid_episode_position(positions,planner,minimun_distance):
    found_match = False
    while not found_match:
        index_start = np.random.randint(len(positions))
        start_pos =positions[index_start]
        if not planner.test_position((start_pos.location.x,start_pos.location.y,22),\
            (start_pos.orientation.x,start_pos.orientation.y,start_pos.orientation.z)):
            continue
        index_goal = np.random.randint(len(positions))
        if index_goal == index_start:
            continue


        print (' TESTING (',index_start,',',index_goal,')')
        goals_pos =positions[index_goal]  
        if not planner.test_position((goals_pos.location.x,goals_pos.location.y,22),\
            (goals_pos.orientation.x,goals_pos.orientation.y,goals_pos.orientation.z)):
            continue
        if sldist([start_pos.location.x,start_pos.location.y],
            [goals_pos.location.x,goals_pos.location.y]) < minimun_distance:

            continue

        if planner.is_there_posible_route((start_pos.location.x,start_pos.location.y,22)\
            ,(start_pos.orientation.x,start_pos.orientation.y,start_pos.orientation.z),\
            (goals_pos.location.x,goals_pos.location.y,22)):
            found_match=True


    return index_start,index_goal


"""


class TestBenchmark(Benchmark):

    def __init__(self, city_name, name_to_save, camera_set, continue_experiment=False):
        self._camera_set = camera_set
        self._planner = Planner(city_name)

        Benchmark.__init__(self, city_name, name_to_save,
                           continue_experiment=continue_experiment)

    def get_all_statistics(self):

        summary = compute_summary(os.path.join(
            self._full_name, self._suffix_name), [3])

        return summary

    def plot_summary_train(self):

        self._plot_summary([1.0, 3.0, 6.0, 8.0])

    def plot_summary_test(self):

        self._plot_summary([4.0, 14.0])

    def _plot_summary(self, weathers):
        """
        We plot the summary of the testing for the set selected weathers.
        The test weathers are [4,14]

        """

        metrics_summary = compute_summary(os.path.join(
            self._full_name, self._suffix_name), [3])

        for metric, values in metrics_summary.items():

            print('Metric : ', metric)
            for weather, tasks in values.items():
                if weather in set(weathers):
                    print('  Weather: ', weather)
                    count = 0
                    for t in tasks:
                        print('    Task ', count, ' -> ', t)
                        count += 1

                    print('    AvG  -> ', float(sum(tasks)) / float(len(tasks)))

    def _calculate_time_out(self, path_distance):
        """
        Function to return the timeout ( in miliseconds) that is calculated based on distance to goal.
        This is the same timeout as used on the CoRL paper.
        """
        return 4
        #return ((path_distance / 100000.0) / 10.0) * 3600.0 + 10.0

    def _poses_town01(self):
        """
        For each matrix is a new task

        """

        def _poses_navigation():
            return [[105, 29]]

        return [_poses_navigation()
                ]

    def _poses_town02(self):

        def _poses_navigation():
            return [[19, 66]]

        return [_poses_navigation()
                ]

    def _build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set the camera
        # This single RGB camera is used on every experiment

        weathers = [1]
        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [20]
            pedestrians_tasks = [50]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [15]
            pedestrians_tasks = [50]

        experiments_vector = []

        for weather in weathers:

            for iteration in range(len(poses_tasks)):

                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather,
                    SeedVehicles=123456789,
                    SeedPedestrians=123456789
                )
                # Add all the cameras that were set for this experiments

                for camera in self._camera_set:
                    conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Id=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector

    def _get_details(self):

        # Function to get automatic information from the experiment for writing purposes
        return 'test_databench_' + self._city_name

    def _get_pose_and_task(self, line_on_file):
        """
        Returns the pose and task this experiment is, based on the line it was
        on the log file.
        """
        # We assume that the number of poses is constant
        return line_on_file / len(self._experiments), line_on_file % 25

