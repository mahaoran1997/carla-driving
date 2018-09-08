import sys

sys.path.append('../configuration')
sys.path.append('../input')
from config import *
from dataset_manager import *


def test_input(gpu_number):
    """ Initialize the input class to get the configuration """
    config = configMain()

    manager = DatasetManager(config.config_input)

    """ Get the batch tensor that is going to be used around """
    batch_tensor = manager.train.get_batch_tensor()
    batch_tensor_val = manager.validation.get_batch_tensor()
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.visible_device_list = gpu_number
    sess = tf.Session(config=config_gpu)
    manager.start_training_queueing(sess)
    manager.start_validation_queueing(sess)

    for i in range(10):

        batch = sess.run(batch_tensor)
        # print "one Image"
        # print batch[0][0]
        count = 1
        for j in range(len(config.targets_names)):
            print(config.targets_names[j])
            print(batch[count][0:20])
            count += 1

        for j in range(len(config.inputs_names)):
            print(config.inputs_names[j])
            print(batch[count][0:20])
            count += 1
