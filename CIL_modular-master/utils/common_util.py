
def parse_drive_arguments(args, driver_conf, attributes):
    for attr in attributes:
        value = getattr(args, attr)
        if value is not None:
            setattr(driver_conf, attr, value)

    if hasattr(args, "port"):
        if args.port is not None:
            driver_conf.port = int(args.port)

    if hasattr(args, "resolution"):
        if args.resolution is not None:
            res_string = args.resolution.split(',')
            driver_conf.resolution = [int(res_string[0]), int(res_string[1])]

    if hasattr(args, "image_cut"):
        if args.image_cut is not None:
            cut_string = args.image_cut.split(',')
            driver_conf.image_cut = [int(cut_string[0]), int(cut_string[1])]

    return driver_conf

import os
import tensorflow as tf

def restore_session(sess, saver, models_path):
    if not os.path.exists(models_path):
        os.mkdir(models_path)
        os.mkdir(models_path + "/train/")
        os.mkdir(models_path + "/val/")

    ckpt = tf.train.get_checkpoint_state(models_path)
    if ckpt:
        print('Restoring from ', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        ckpt = 0

    return ckpt

import scipy.misc

def preprocess_image(sensor, image_cut, sensor_size):
    sensor = sensor[image_cut[0]:image_cut[1], :, :3]
    sensor = sensor[:, :, ::-1]
    if sensor_size is not None:
        sensor = scipy.misc.imresize(sensor, [sensor_size[0], sensor_size[1]])

    return sensor
