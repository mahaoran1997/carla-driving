#!/usr/bin/env bash

# resource related
gpu_carla="0"
gpu_agent="3"
port="2003"
# test related
city_name="Town02" # first test the train town
exp_id="mm45_v4_from_alldata_perception_v9_nodrop"


# launch carla
export CUDA_VISIBLE_DEVICES=$gpu_carla
#/scratch/yang/aws_data/carla_0.8.4/CarlaUE4.sh /Game/Maps/$city_name -carla-server -carla-world-port=$port &
#sleep 20

# launch the client
export CUDA_VISIBLE_DEVICES=$gpu_agent
ulimit -Sn 60000

python \
drive_interfaces/carla/comercial_cars/run_test_cvpr.py \
-e $exp_id \
-s $exp_id \
-l 127.0.0.1 \
-p $port \
-cy $city_name \
-m 0.1 || true #TestTownTrainWeather

