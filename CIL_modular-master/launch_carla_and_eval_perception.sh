#!/usr/bin/env bash

# resource related
gpu_agent="3"
exp_id="mm45_v5_perception_allpercep_nowd"
# end of resource



declare -a arr=("Town02" "Town01")
for city_name in "${arr[@]}"
do
    port=$(python get_unused_port.py)
    echo "using port "$port
    # launch carla
    /scratch/yang/aws_data/carla_0.8.4/CarlaUE4.sh /Game/Maps/$city_name -carla-server -carla-settings="/scratch/yang/aws_data/carla_0.8.4/carla_long_timeout.ini" -carla-world-port=$port &
    pid_carla=$!
    sleep 20

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
    -m 0.1 &

    #pkill -9 -P $pid_carla
done