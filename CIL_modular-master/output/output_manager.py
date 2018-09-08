"""Visualization libs"""
import sys
import tensorflow as tf
sys.path.append('../utils')

from validation_manager import ValidationManager
from codification import *


def convert_mat_to_tensor(py_mat, branch_config):
    tensor_vec = []
    for i in range(len(py_mat)):
        tensor_vec.append(tf.split(tf.reduce_mean(tf.convert_to_tensor(py_mat[i]), axis=[1]), len(branch_config[i])))

    return tensor_vec


class OutputManager(object):
    def __init__(self, config, training_manager, config_train, sess, batch_tensor_val):
        self._config = config
        self._training_manager = training_manager
        self._sess = sess

        self.tensorboard_scalars()
        self.tensorboard_images()
        self._merged = tf.summary.merge_all()

        self._validater = ValidationManager(config, training_manager, sess, batch_tensor_val, self._merged)

        self.first_time = True
        self.duration_sum = 0.0

    def tensorboard_scalars(self):
        tf.summary.scalar('Loss', tf.reduce_mean(self._training_manager.get_loss()))

        tf.summary.scalar('learning_rate', tf.reduce_mean(self._training_manager._variable_learning))

        """ This is the loss energy vec """
        # indexed by [ibranch][i_within_branch]
        energy_tensor_vec = convert_mat_to_tensor(self._training_manager.get_variable_energy(),
                                                  self._config.branch_config)
        for i in range(len(energy_tensor_vec)):
            for j in range(len(self._config.branch_config[i])):
                tf.summary.scalar('Energy_B_' + str(i) + '_' + self._config.branch_config[i][j],
                                  tf.squeeze(energy_tensor_vec[i][j]))

        variables_tensor_vec = convert_mat_to_tensor(self._training_manager.get_variable_error(),
                                                     self._config.branch_config)
        for i in range(len(variables_tensor_vec)):
            for j in range(len(self._config.branch_config[i])):
                tf.summary.scalar('Error_B_' + str(i) + '_' + self._config.branch_config[i][j],
                                  tf.squeeze(variables_tensor_vec[i][j]))

        for i in range(len(self._config.targets_names)):
            tf.summary.histogram('GT_B_' + str(i) + '_' + self._config.targets_names[i],
                                 self._training_manager._targets_data[i])

        for i in range(len(variables_tensor_vec)):
            for j in range(len(self._config.branch_config[i])):  # for other branches
                tf.summary.histogram('Output_B_' + str(i) + '_' + self._config.branch_config[i][j],
                                     self._training_manager._output_network[i][:, j])

        self._train_writer = tf.summary.FileWriter(self._config.train_path_write, self._sess.graph)

    def tensorboard_images(self):
        if self._config.segmentation_model != None:
            if not self._config.use_perception_stack:
                tf.summary.image('Image_input', self._training_manager._input_images)
            #tf.summary.image('Image_vbp', self._training_manager._vis_images)
            tf.summary.image('Segmentation_output', self._training_manager._gray)
        else:
            if not self._config.use_perception_stack:
                tf.summary.image('Image_input', self._training_manager._input_images)
            #tf.summary.image('Image_vbp', self._training_manager._vis_images)

    def write_tensorboard_summary(self, i):
        feedDict = self._training_manager.get_feed_dict()
        feedDict[self._training_manager._dout] = [1.0] * len(self._config.dropout)
        summary = self._sess.run(self._merged, feed_dict=feedDict)

        self._train_writer.add_summary(summary, i)

    def print_outputs(self, i, duration):
        self.duration_sum += duration
        # the dictonary of the data used for training
        if i % self._config.print_interval == 0:
            # ideally also include: Epoch, sample inputs and targets
            # print("step=%d, images/second=%f, train loss=%f, validation loss=%f\n" % (i, 1.0*self._config/duration, 0.0, 0.0))
            print("step=%d, images/second=%f" % (i, 1.0 * self._config.batch_size / self.duration_sum * self._config.print_interval))
            self.duration_sum = 0.0

        """ Writing summary """
        if i % self._config.summary_writing_period == 0 or self.first_time:
            self.first_time = False
            self.write_tensorboard_summary(i)

        if i % self._config.validation_period == 0:
            self._validater.run(i)
