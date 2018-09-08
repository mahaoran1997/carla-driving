#!/usr/bin/env bash

gpu_agent=$1
gpu_carla=$2
gpu_perception=$3
weathers=$4
exp_id=$5
city_name=$6


port=$(python get_unused_port.py)
echo "using port "$port
# launch carla
/scratch/yang/aws_data/carla_0.8.4/CarlaUE4.sh /Game/Maps/$city_name -carla-server -carla-settings="../scratch/carla_0.8.4/carla_long_timeout.ini" -carla-world-port=$port &
pid_carla=$!
sleep 20

# launch the client
export CUDA_VISIBLE_DEVICES=$gpu_agent
ulimit -Sn 60000

python \
drive_interfaces/carla/comercial_cars/run_test_cvpr.py \
-e $exp_id \
-s $exp_id"_"$weathers \
-l 127.0.0.1 \
-p $port \
-cy $city_name \
-m 0.05 \
--weathers $weathers \
--benchmark_name "YangExp3cam" \
--gpu_perceptions $gpu_perception

pkill -9 -P $pid_carla
