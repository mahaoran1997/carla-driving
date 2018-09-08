import tensorflow as tf


def one_hot_to_image(x):
    print('ONE HOT TO IMAGE')
    print(x)
    number_labels = x.get_shape().as_list()[3]
    print(('Number of Labels:', number_labels))
    # classes_values =tf.constant([x*int(255.0/(float(number_labels)-1)) for x in range(number_labels) ])

    prediction = tf.argmax(x, -1)
    print(prediction)
    gray_x = tf.scalar_mul(int(255.0 / (float(number_labels) - 1)), prediction)

    # multiplied_classes = tf.multiply(classes_values,x)
    # print multiplied_classes
    # gray_x = tf.reduce_max(multiplied_classes,axis=3,keep_dims=True)
    print(gray_x)
    gray_x = tf.cast(gray_x, dtype=tf.float32)
    gray_x = tf.scalar_mul(1.0 / 255.0, gray_x)

    print(gray_x)
    return gray_x


def image_to_one_hot(x, number_labels):
    # multiply the number_labels and then convert to one hot
    # the original x should be between 0 and 1.0, otherwise not make sense
    print('IMAGE TO ONE HOT')
    print(x)

    one_hot = tf.squeeze(tf.scalar_mul(number_labels, x))
    print(one_hot)

    one_hot = tf.cast(one_hot, dtype=tf.int32)
    one_hot = tf.one_hot(one_hot, number_labels)
    print(one_hot)

    one_hot = tf.reshape(one_hot, [-1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], number_labels])

    return one_hot


def label_to_one_hot(x, number_labels):
    # directly to one hot, without multiplying
    print('LABEL TO ONE HOT')
    print(x)

    one_hot = tf.squeeze(x)
    print(one_hot)

    one_hot = tf.cast(one_hot, dtype=tf.int32)
    one_hot = tf.one_hot(one_hot, number_labels)
    print(one_hot)

    one_hot = tf.reshape(one_hot, [-1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], number_labels])

    return one_hot


def weight_variable_scaling(shape, name):
    initializer = tf.uniform_unit_scaling_initializer(factor=0.8)
    initial = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=True)
    return initial


def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)
    # return tf.get_variable(name=name,initializer= tf.constant(1.0,shape=shape),trainable=False)


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    # initial = tf.constant(0.1, shape=shape,name=name)
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)
    # initial = tf.constant(0.1, shape=shape,name=name)
    # return tf.get_variable(name=name,initializer= tf.constant(0.1,shape=shape))


class Network(object):
    def __init__(self, config, dropout, image_shape):

        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._config = config
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):

        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, name='W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name='max_pool' + str(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=self._config.is_training, updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):

        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def gated_block(self, x, h, output_size):

        self._count_fc += 1

        filters_in_x = x.get_shape()[-1]
        filters_in_h = h.get_shape()[-1]

        print(x)
        print(h)
        print(output_size)
        print(filters_in_x)

        x_vec = tf.split(0, self._config.batch_size, x)
        h_vec = tf.split(0, self._config.batch_size, h)
        y_vec = []
        print(x_vec)
        for i in range(0, len(x_vec)):
            y_vec.append(tf.reshape(tf.matmul(tf.transpose(x_vec[i]), h_vec[i]), [1, -1]))

        y = tf.concat(0, y_vec)

        # Probably we have to resize the y
        y_shape = y.get_shape()[-1]
        shape = [y_shape, output_size]

        print(y)
        print(shape)

        weights = weight_xavi_init(shape, 'W_fc_gate' + str(self._count_fc))
        bias = bias_variable([output_size], name='W_b_fc_gate' + str(self._count_fc))

        return tf.nn.xw_plus_b(y, weights, bias, name='fc' + str(self._count_fc))

    def soft_max(self, x):
        self._count_soft_max += 1
        x = tf.nn.softmax(x, name='soft' + str(self._count_soft_max))
        self._features['soft_block' + str(self._count_soft_max)] = x
        return x

    def lstm(self, x):
        self._count_lstm += 1

        filters_in = x[0].get_shape()[-1]

        lstm_cells = tf.nn.rnn_cell.LSTMCell(int(filters_in), forget_bias=1.0)

        """ For 200 images for example, with a sequence size of 50 , there are four sequences being trained here """

        outputs, states_1 = tf.nn.rnn(lstm_cells, x, dtype=tf.float32, scope='lstm' + str(self._count_lstm))

        return outputs

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)

            return self.activation(x)

    def conv_block_nobn(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.dropout(x)

            return self.activation(x)

    def fc_block(self, x, output_size):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def residual_block(self, x, stride, out_filter, activate_before_residual=False, padding_in='SAME'):
        """Residual unit with 2 sub layers."""
        with tf.name_scope("res_block" + str(self._count_conv)):
            in_filter = x.get_shape().as_list()[-1]
            if activate_before_residual:
                with tf.variable_scope('shared_activation'):
                    x = self.bn(x)
                    x = self.activation(x)
                    orig_x = x
            else:
                with tf.variable_scope('residual_only_activation'):
                    orig_x = x
                    x = self.bn(x)
                    x = self.activation(x)

            with tf.variable_scope('sub1'):
                x = self.conv(x, 3, stride, out_filter, padding_in=padding_in)

            with tf.variable_scope('sub2'):
                x = self.bn(x)
                x = self.activation(x)
                x = self.conv(x, 3, 1, out_filter, padding_in=padding_in)

            with tf.variable_scope('sub_add'):
                if in_filter != out_filter:
                    orig_x = tf.nn.avg_pool(orig_x, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')

                    orig_x = tf.pad(
                        orig_x, [[0, 0], [0, 0], [0, 0],
                                 [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])

                x += orig_x

        return x

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features

        self._count_lstm += 1

        filters_in = x[0].get_shape()[-1]

        lstm_cells = tf.nn.rnn_cell.LSTMCell(int(filters_in), forget_bias=1.0)

        """ For 200 images for example, with a sequence size of 50 , there are four sequences being trained here """

        outputs, states_1 = tf.nn.rnn(lstm_cells, x, dtype=tf.float32, scope='lstm' + str(self._count_lstm))

        return outputs

    def get_vbp_images(self, xc):

        with tf.name_scope('vbp'):
            for i in reversed(list(range(self._count_conv))):  # reversely go through the feature maps

                if i == self._count_conv - 1:
                    feature_map = xc
                else:
                    feature_map = self._features['conv_block' + str(i)]  # Get this one

                print(feature_map)
                feature_map = tf.reduce_mean(feature_map, 3, keep_dims=True)  # Apply average of all feature maps.
                print(feature_map)

                print(i)
                print(len(self._conv_kernels))
                print(self._conv_kernels)
                print(self._conv_strides)
                if i != 0:
                    next_shape = tf.shape(self._features['conv_block' + str(i - 1)])
                else:
                    next_shape = self._image_shape
                batch_size = next_shape[0]
                shape1 = next_shape[1]
                shape2 = next_shape[2]
                output_shape_tensor = tf.convert_to_tensor([batch_size, shape1, shape2, 1])

                deconv_weights = weight_ones([self._conv_kernels[i], self._conv_kernels[i], 1, 1],
                                             'const_deconv' + str(i))

                print(i)
                if i == self._count_conv - 1:
                    feature_map_up = tf.nn.conv2d_transpose(feature_map, deconv_weights, \
                                                            output_shape_tensor,
                                                            [1, self._conv_strides[i], self._conv_strides[i], 1], \
                                                            padding='VALID',
                                                            name='deconv' + str(i))  # apply a deconvolution

                    vbp_image = feature_map_up
                else:

                    vbp_image = tf.multiply(vbp_image, feature_map)  # multiply with the last one
                    vbp_image = tf.nn.conv2d_transpose(vbp_image, deconv_weights, \
                                                       output_shape_tensor,
                                                       [1, self._conv_strides[i], self._conv_strides[i], 1], \
                                                       padding='VALID', name='deconv' + str(i))  # apply a deconvolution

                print(vbp_image)
        return vbp_image
