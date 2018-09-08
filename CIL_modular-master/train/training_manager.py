import sys, time, os
import tensorflow as tf
import tensorflow.contrib.slim as slim
sys.path.append('train')
sys.path.append('utils')
sys.path.append('structures')

import loss_functions
from enet import *
from erfnet import *
from network import one_hot_to_image, image_to_one_hot, label_to_one_hot

def save_model(saver, sess, models_path, i):
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    saver.save(sess, models_path + '/model.ckpt', global_step=i)
    print(('Model saved at iteration:', i))


def get_last_iteration(ckpt):
    if ckpt:
        return int(ckpt.model_checkpoint_path.split('-')[1])
    else:
        return 1

# with the name of TrainManager, it actually is a Network manager
class TrainManager(object):
    def __init__(self, config, reuse, placeholder_input=True, batch_tensor=None):
        # TODO: update the initializer
        self._config = config
        self._reuse = reuse
        self._placeholder_input = placeholder_input

        with tf.device('/gpu:0'):
            if placeholder_input:
                self._input_images = tf.placeholder(tf.float32, shape=[None,
                                                                    config.feature_input_size[0],
                                                                    config.feature_input_size[1],
                                                                    config.feature_input_size[2]], name="input_image")
                self._targets_data = []
                for i in range(len(self._config.targets_names)):
                    self._targets_data.append(tf.placeholder(tf.float32, shape=[None, self._config.targets_sizes[i]],
                                                             name="target_" + self._config.targets_names[i]))
                self._input_data = []
                for i in range(len(self._config.inputs_names)):
                    self._input_data.append(tf.placeholder(tf.float32, shape=[None, self._config.inputs_sizes[i]],
                                                           name="input_" + self._config.inputs_names[i]))
            else:
                self._input_images = batch_tensor[0]
                self._targets_data = batch_tensor[1:(1+len(self._config.targets_names))]
                self._input_data   = batch_tensor[(1+len(self._config.targets_names)):]

            self._dout = tf.placeholder("float", shape=[len(config.dropout)])
            self._variable_learning = tf.placeholder("float", name="learning")

        self._feedDict = {}

        self._create_structure = __import__(config.network_name).create_structure
        self._loss_function = getattr(loss_functions, config.loss_function)  # The function to call


    def build_network(self):
        """ Depends on the actual input """
        with tf.name_scope("Network"):
            self._output_network, self._vis_images, self._features, self._weights = self._create_structure(tf,
                                                                                                           self._input_images,
                                                                                                           self._input_data,
                                                                                                           self._config.image_size,
                                                                                                           self._dout,
                                                                                                           self._config)

    def build_seg_network_erfnet_one_hot(self):
        """ Depends on the actual input """
        self._seg_network = ErfNet_Small(self._input_images[:, :, :, 0:3], self._config.number_of_labels,
                                         batch_size=self._config.batch_size, reuse=self._reuse,
                                         is_training=self._config.train_segmentation)[0]
        with tf.name_scope("Network"):
            # with tf.variable_scope("Network",reuse=self._reuse):
            # print  self._seg_network

            self._sensor_input = self._seg_network
            # Just for visualization
            self._gray = one_hot_to_image(self._seg_network)
            self._gray = tf.expand_dims(self._gray, -1)

            self._output_network, self._vis_images, self._features, self._weights \
                = self._create_structure(tf, self._sensor_input, self._input_data, self._config.image_size, self._dout,
                                         self._config)

    def build_loss(self):
        with tf.name_scope("Loss"):
            self._loss, self._variable_error, self._variable_energy, self._image_loss, self._branch \
                = self._loss_function(self._output_network,
                                      self._targets_data,
                                      self._input_data[self._config.inputs_names.index("Control")],
                                      self._config)

    def build_optimization(self):
        """ List of Interesting Parameters """
        #		beta1=0.7,beta2=0.85
        #		beta1=0.99,beta2=0.999
        with tf.name_scope("Optimization"):
            print("using optimizer ", self._config.optimizer)
            if self._config.optimizer == "sgd":
                opt = tf.train.MomentumOptimizer
                opt_kwargs = {"momentum": self._config.momentum}
            elif self._config.optimizer == "adam":
                opt = tf.train.AdamOptimizer
                opt_kwargs = {}
            else:
                raise ValueError()

            if hasattr(self._config, 'finetune_segmentation') or \
                    not (hasattr(self._config, 'segmentation_model_name')) or \
                    self._config.segmentation_model is None:
                self._train_step = opt(self._variable_learning, **opt_kwargs).minimize(self._loss)
                print("Optimizer: All variables")
            else:
                train_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) -
                                  set(slim.get_variables(scope=str(self._config.segmentation_model_name))))
                self._train_step = opt(self._variable_learning, **opt_kwargs).minimize(self._loss, var_list=train_vars)
                print("Optimizer: Exclude variables from: ", str(self._config.segmentation_model_name))

    def run_train_step(self, batch_tensor, sess, i):
        # TODO: make sure no one use batch_tensor for val
        capture_time = time.time()

        # Get the change in the learning rate]
        # let the default decrease factor to be the last one.
        schedule = [[0, 1.0]] + self._config.training_schedule
        decrease_factor = schedule[-1][1]
        for j in range(len(schedule)):
            if i < schedule[j][0]:
                decrease_factor = schedule[j-1][1]
                break
        self._feedDict = {self._variable_learning: decrease_factor * self._config.learning_rate,
                          self._dout: self._config.dropout}

        assert(self._placeholder_input == False)

        sess.run(self._train_step, feed_dict=self._feedDict)

        return time.time() - capture_time

    def get_variable_energy(self):
        return self._variable_energy

    def get_loss(self):
        return self._loss

    def get_variable_error(self):
        return self._variable_error

    def get_feed_dict(self):
        return self._feedDict
