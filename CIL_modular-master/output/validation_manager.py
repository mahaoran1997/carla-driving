"""Visualization libs"""
import sys
import tensorflow as tf

sys.path.append('../utils')
from codification import *

class ValidationManager(object):
    def __init__(self, config, training_manager, sess, batch_tensor, merged_summary):
        self._training_manager = training_manager
        self._sess = sess
        self._config = config
        self._batch_tensor = batch_tensor
        self._merged = merged_summary
        self._val_writer = tf.summary.FileWriter(self._config.val_path_write, self._sess.graph)

    def load_dict(self, batch):
        feedDict = {self._training_manager._input_images: batch[0]}

        count = 1
        for i in range(len(self._config.targets_names)):
            feedDict.update({self._training_manager._targets_data[i]: batch[count]})
            count += 1

        for i in range(len(self._config.inputs_names)):
            feedDict.update({self._training_manager._input_data[i]: batch[count]})
            count += 1
        feedDict.update({self._training_manager._dout: [1] * len(self._config.dropout)})

        feedDict.update({self._training_manager._variable_learning: 0.0})

        return feedDict

    def run(self, iter_number):
        number_of_batches = self._config.number_images_val // self._config.batch_size_val
        assert (self._config.number_images_val % self._config.batch_size_val == 0)

        sumEnergy = 0.0
        for j in range(0, number_of_batches):
            batch_val = self._sess.run(self._batch_tensor)
            feedDictVal = self.load_dict(batch_val)

            if j == 0:
                summary = self._sess.run(self._merged, feed_dict=feedDictVal)
                self._val_writer.add_summary(summary, iter_number)

            energy_val = self._sess.run(self._training_manager.get_loss(), feed_dict=feedDictVal)

            sumEnergy += sum(energy_val)

        with open(self._config.val_path_write + 'loss_function_val', 'a+') as outfile_energy:
            outfile_energy.write("iter %d, val loss %f\n" % (iter_number, sumEnergy/self._config.number_images_val))
