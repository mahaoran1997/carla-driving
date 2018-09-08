#!/usr/bin/env bash

# you could switch the carla config through the command line, thus it's easy to loop over all carla config files.

# some pending flags
# --carla-config
# --show_screen
# --noise

python chauffeur.py \
    drive \
    --gpu 1 \
    --log \
    --debug \
    --path "./data" \
    --driver-config "9cam_agent_carla_acquire_rc_yang_machine" \
    --experiment-name "mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_yang"
