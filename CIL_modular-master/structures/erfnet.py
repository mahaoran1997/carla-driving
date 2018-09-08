import tensorflow as tf

slim = tf.contrib.slim

'''
=============================================================
ErfNet: Efficient ConvNet for Real-time Semantic Segmentation
=============================================================
Based on the paper: http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf
'''


def spatial_dropout(x, p, seed, scope, is_training=True):
    '''
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input feature map.
    Note that p stands for the probability of dropping, but tf.nn.relu uses probability of keeping.

    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------

    INPUTS:
    - x(Tensor): a 4D Tensor of the input feature map.
    - p(float): a float representing the probability of dropping a layer
    - seed(int): an integer for random seeding the random_uniform distribution that runs under tf.nn.relu
    - scope(str): the string name for naming the spatial_dropout
    - is_training(bool): to turn on dropout only when training. Optional.

    OUTPUTS:
    - output(Tensor): a 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    '''
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x


@slim.add_arg_scope
def filter_scale(inputs, output_depth, is_training=True, scope='filter_scale'):
    net = slim.conv2d(inputs, output_depth, [3, 3], activation_fn=None, scope=scope + '_conv')
    net = slim.batch_norm(net, is_training=is_training, fused=None, scope=scope + '_batch_norm')
    net = tf.nn.relu(net, name=scope + '_relu')

    return net


@slim.add_arg_scope
def downsampler(inputs, output_depth, is_training=True, scope='downsampler'):
    '''
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    - output_depth(int): an integer indicating the output depth of the output convolutional block.

    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    input_shape = inputs.get_shape().as_list()
    input_depth = input_shape[-1]

    # Convolutional branch
    net_conv = slim.conv2d(inputs, output_depth - input_depth, [3, 3], stride=2, activation_fn=None,
                           scope=scope + '_conv')

    # Max pool branch
    net_pool = slim.max_pool2d(inputs, [2, 2], stride=2, scope=scope + '_max_pool')

    # Concatenated output
    net = tf.concat([net_conv, net_pool], axis=3, name=scope + '_concat')

    net = slim.batch_norm(net, is_training=is_training, fused=None, scope=scope + '_batch_norm')
    net = tf.nn.relu(net, name=scope + '_relu')

    return net


@slim.add_arg_scope
def upsampler(inputs, output_depth, is_training=True, scope='upsampler'):
    '''
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    - output_depth(int): an integer indicating the output depth of the output convolutional block.

    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''

    # Deconvolution
    net_conv = slim.conv2d_transpose(inputs, output_depth, [3, 3], stride=2, activation_fn=None,
                                     scope=scope + '_deconv')
    net = slim.batch_norm(net_conv, is_training=is_training, fused=None, scope=scope + '_batch_norm')
    net = tf.nn.relu(net, name=scope + '_relu')

    return net


@slim.add_arg_scope
def non_bottleneck(inputs,
                   output_depth,
                   filter_size=3,
                   regularizer_prob=0.3,
                   seed=0,
                   is_training=True,
                   dilation_rate=None,
                   scope='non_bottleneck'):
    '''
    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape [batch_size, height, width, num_channels].
    - output_depth(int): an integer indicating the output depth of the output convolutional block.
    - filter_size(int): an integer that gives the height and width of the filter size to use for a regular/dilated convolution.
    - regularizer_prob(float): the float p that represents the prob of dropping a layer for spatial dropout regularization.
    - seed(int): an integer for the random seed used in the random normal distribution within dropout.
    - is_training(bool): a boolean value to indicate whether or not is training. Decides batch_norm and prelu activity.
    - dilation_rate(int): the dilation factor for performing atrous convolution/dilated convolution.
    - scope(str): a string name that names your bottleneck.

    OUTPUTS:
    - net(Tensor): The convolution block output after a bottleneck
    - pooling_indices(Tensor): If downsample, then this tensor is produced for use in upooling later.
    - inputs_shape(list): The shape of the input to the downsampling conv block. For use in unpooling later.

    '''

    # ============NON-BOTTLENECK 1D====================
    # Check if dilation rate is given
    if not dilation_rate:
        raise ValueError('Dilation rate is not given.')

    # Save the main branch for addition later
    net_main = inputs

    # First conv block - asymmetric convolution
    net = slim.conv2d(inputs, output_depth, [filter_size, 1], scope=scope + '_conv1a')
    net = tf.nn.relu(net, name=scope + '_relu1a')
    net = slim.conv2d(net, output_depth, [1, filter_size], scope=scope + '_conv1b')
    net = slim.batch_norm(net, is_training=is_training, fused=None, scope=scope + '_batch_norm1')
    net = tf.nn.relu(net, name=scope + '_relu1b')

    # Second conv block - asymmetric + dilation convolution
    net = slim.conv2d(net, output_depth, [filter_size, 1], rate=[dilation_rate, 1], scope=scope + '_conv2a')
    net = tf.nn.relu(net, name=scope + '_relu2a')
    net = slim.conv2d(net, output_depth, [1, filter_size], rate=[1, dilation_rate], scope=scope + '_conv2b')
    net = slim.batch_norm(net, is_training=is_training, fused=None, scope=scope + '_batch_norm2')

    # Regularizer
    net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope + '_spatial_dropout')

    # Add the main branch
    net = tf.add(net_main, net, name=scope + '_add_residual')
    net = tf.nn.relu(net, name=scope + '_relu2b')

    return net


# Now actually start building the network
def ErfNet(inputs,
           num_classes,
           batch_size=12,
           reuse=None,
           is_training=True,
           scope='ErfNet'):
    '''
    The ErfNet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - batch_size(int): the batch size to explictly set the shape of the inputs in order for operations to work properly.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''
    # Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    print(inputs.shape)

    with tf.variable_scope(scope, reuse=reuse):
        # Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([downsampler, upsampler, non_bottleneck], is_training=is_training), \
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):
            # =================START=================
            net = downsampler(inputs, output_depth=16, is_training=True, scope='downsampler_1')
            net = downsampler(net, output_depth=64, is_training=True, scope='downsampler_2')

            for i in range(3, 8):  # 5 times
                net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=1, regularizer_prob=0.03,
                                     scope='non_bottleneck_' + str(i))

            net = downsampler(net, output_depth=128, is_training=True, scope='downsampler_8')

            for i in range(0, 2):  # 2 times
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=2, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(9 + i * 4))
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=4, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(10 + i * 4))
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=8, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(11 + i * 4))
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=16, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(12 + i * 4))

            net = upsampler(net, output_depth=64, is_training=True, scope='upsampler_17')
            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_18')
            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_19')

            net = upsampler(net, output_depth=16, is_training=True, scope='upsampler_20')
            net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_21')
            net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_22')

            logits = upsampler(net, output_depth=num_classes, is_training=True, scope='upsampler_23')

            # =============END=============

            probabilities = tf.nn.softmax(logits, name='logits_to_softmax')

        return logits, probabilities


def ErfNet_Small(inputs,
                 num_classes,
                 batch_size,
                 reuse=None,
                 is_training=True,
                 scope='ErfNet_Small'):
    '''
    The ErfNet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - batch_size(int): the batch size to explictly set the shape of the inputs in order for operations to work properly.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''
    # Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    with tf.variable_scope(scope, reuse=reuse):
        # Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([downsampler, upsampler, non_bottleneck], is_training=is_training), \
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):
            # =================START=================
            net = downsampler(inputs, output_depth=16, is_training=True, scope='downsampler_1')

            for i in range(2, 7):  # 5 times
                net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0.03,
                                     scope='non_bottleneck_' + str(i))

            net = downsampler(net, output_depth=64, is_training=True, scope='downsampler_7')

            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=2, regularizer_prob=0.3,
                                 scope='non_bottleneck_8')
            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=4, regularizer_prob=0.3,
                                 scope='non_bottleneck_9')
            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=8, regularizer_prob=0.3,
                                 scope='non_bottleneck_10')
            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=16, regularizer_prob=0.3,
                                 scope='non_bottleneck_11')

            net = upsampler(net, output_depth=16, is_training=True, scope='upsampler_12')
            net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_13')
            net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_14')

            logits = upsampler(net, output_depth=num_classes, is_training=True, scope='upsampler_15')

            # =============END=============

            probabilities = tf.nn.softmax(logits, name='logits_to_softmax')

        return logits, probabilities


def ErfNet_NoDS(inputs,
                num_classes,
                batch_size,
                reuse=None,
                is_training=True,
                scope='ErfNet'):
    '''
    The ErfNet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - batch_size(int): the batch size to explictly set the shape of the inputs in order for operations to work properly.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''
    # Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    with tf.variable_scope(scope, reuse=reuse):
        # Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([downsampler, upsampler, non_bottleneck], is_training=is_training), \
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):
            # =================START=================
            # net = downsampler(inputs, output_depth=16, is_training=True, scope='downsampler_1')
            # net = downsampler(net, output_depth=64, is_training=True, scope='downsampler_2')
            net = filter_scale(inputs, output_depth=16, is_training=True, scope='filter_scale_1')
            net = filter_scale(net, output_depth=64, is_training=True, scope='filter_scale_2')

            for i in range(3, 8):  # 5 times
                net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=1, regularizer_prob=0.03,
                                     scope='non_bottleneck_' + str(i))

            # net = downsampler(net, output_depth=128, is_training=True, scope='downsampler_8')
            net = filter_scale(net, output_depth=128, is_training=True, scope='filter_scale_8')

            for i in range(0, 2):  # 2 times
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=2, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(9 + i * 4))
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=4, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(10 + i * 4))
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=8, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(11 + i * 4))
                net = non_bottleneck(net, output_depth=128, filter_size=3, dilation_rate=16, regularizer_prob=0.3,
                                     scope='non_bottleneck_' + str(12 + i * 4))

            # net = upsampler(net, output_depth=64, is_training=True, scope='upsampler_17')
            net = filter_scale(net, output_depth=64, is_training=True, scope='filter_scale_17')

            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_18')
            net = non_bottleneck(net, output_depth=64, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_19')

            # net = upsampler(net, output_depth=16, is_training=True, scope='upsampler_20')
            net = filter_scale(net, output_depth=16, is_training=True, scope='filter_scale_20')
            net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_21')
            net = non_bottleneck(net, output_depth=16, filter_size=3, dilation_rate=1, regularizer_prob=0,
                                 scope='non_bottleneck_22')

            # logits = upsampler(net, output_depth=num_classes, is_training=True, scope='upsampler_23')
            logits = filter_scale(net, output_depth=num_classes, is_training=True, scope='filter_scale_23')

            # =============END=============

            probabilities = tf.nn.softmax(logits, name='logits_to_softmax')

        return logits, probabilities
