import numpy as np
from network import Network
import tensorflow as tf

def depth2space(x, factor):
    shape = x.get_shape()
    assert(shape[3] % (factor**2) == 0)

    x = tf.reshape(x, [-1, shape[1], shape[2], shape[3]//factor//factor, factor, factor])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, [-1, shape[1] * factor, shape[2] * factor, shape[3]//factor//factor])
    return x


def visualize_index(pred):
    # 19*3
    color = np.array([[0, 0, 142], [128, 64,128], [244, 35,232], [ 70, 70, 70], [70,130,180], [230,150,140]], dtype=np.uint8)

    shape = pred.shape

    pred = pred.argmax(axis=3)
    pred = pred.ravel()
    pred = np.array([color[pred, 0], color[pred, 1], color[pred, 2]])
    pred = np.transpose(pred)
    pred = pred.reshape(shape[0], shape[1], shape[2], 3)

    return pred.astype(np.uint8)


def create_structure(tf, input_image, input_data, input_size, dropout, config):
    branches = []

    x = input_image
    # assert the input image is the segmentation output, which has size None*39*52*54
    x = depth2space(x, 3)
    # now it has shape [None, 39*3, 52*3, 9]
    # now we add it's visualization
    xshape = x.get_shape()
    vis = tf.py_func(visualize_index, [x], tf.uint8)
    vis.set_shape([xshape[0], xshape[1], xshape[2], 3])
    tf.summary.image('SegVis', vis)


    network_manager = Network(config, dropout, tf.shape(x))

    '''dimension reduction'''
    # size 39*52*295
    # new shape 39*3 = 117, 52*3 = 156
    xc = network_manager.conv_block(x, 3, 2, 64, padding_in='VALID')
    # size 58*77
    print(xc)
    """conv1"""  # kernel sz, stride, num feature maps
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    # size 56*75
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    # size 27*37
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    # size 25*35
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 256, padding_in='VALID')
    # size 12*17
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    # size 10*15
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 2, 512, padding_in='VALID')
    # size 4*7
    print(xc)
    xc = tf.reduce_mean(xc, axis=[1, 2], keep_dims=True)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x)

    """ fc1 """
    x = network_manager.fc_block(x, 512)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512)

    """Process Control"""
    # control = tf.reshape(control, [-1, int(np.prod(control.get_shape()[1:]))],name = 'reshape_control')
    # print control

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[config.inputs_names.index("Speed")]  # get the speed from input data
        speed = network_manager.fc_block(speed, 128)
        speed = network_manager.fc_block(speed, 128)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512)

    """Start BRANCHING"""
    for i in range(0, len(config.branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if config.branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256)
                branch_output = network_manager.fc_block(branch_output, 256)
            else:
                branch_output = network_manager.fc_block(j, 256)
                branch_output = network_manager.fc_block(branch_output, 256)

            branches.append(network_manager.fc(branch_output, len(config.branch_config[i])))

        print(branch_output)

    weights = network_manager.get_weigths_dict()

    features = network_manager.get_feat_tensors_dict()

    vis_images = network_manager.get_vbp_images(xc)
    print(vis_images)

    print(vis_images.get_shape())

    # vis_images = tf.div(vis_images  -tf.reduce_min(vis_images),tf.reduce_max(vis_images) -tf.reduce_min(vis_images))

    # branches: each of them is a vector of the output(all vars you care) conditioned on that input control signal
    return branches, vis_images, features, weights
