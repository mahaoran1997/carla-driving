import numpy as np

from network import Network


def create_structure(tf, input_image, input_data, input_size, dropout, config):
    branches = []

    x = input_image

    network_manager = Network(config, dropout, tf.shape(x))

    """conv1"""  # kernel sz, stride, num feature maps
    xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='SAME')
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='SAME')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='SAME')
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='SAME')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='SAME')
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 2, 256, padding_in='SAME')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
    print(xc)
    """conv5"""
    xc = network_manager.conv_block(xc, 3, 2, 512, padding_in='SAME')
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

    """Process Control"""
    # control = tf.reshape(control, [-1, int(np.prod(control.get_shape()[1:]))],name = 'reshape_control')
    # print control

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[config.inputs_names.index("Speed")]  # get the speed from input data
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

    '''
    vis_images = network_manager.get_vbp_images(xc)
    print(vis_images)

    print(vis_images.get_shape())
    '''

    # vis_images = tf.div(vis_images  -tf.reduce_min(vis_images),tf.reduce_max(vis_images) -tf.reduce_min(vis_images))

    # branches: each of them is a vector of the output(all vars you care) conditioned on that input control signal
    return branches, None, features, weights
