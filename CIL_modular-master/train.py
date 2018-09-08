import sys, time

sys.path.append('configuration')
sys.path.append('input')
sys.path.append('train')
sys.path.append('output')
sys.path.append('utils')
sys.path.append('../')

from common_util import restore_session

from dataset_manager import *
from training_manager import TrainManager, get_last_iteration, save_model
from output_manager import OutputManager

from all_perceptions import Perceptions

import tensorflow as tf
slim = tf.contrib.slim


def train(experiment_name, memory_fraction):
    """ Initialize the input class to get the configuration """
    conf_module = __import__(experiment_name)
    config_main = conf_module.configMain()
    config_input = conf_module.configInput()

    if config_input.use_perception_stack:
        use_mode = {}
        for key in config_input.perception_num_replicates:
            if config_input.perception_num_replicates[key] > 0:
                assert (config_input.batch_size % config_input.perception_batch_sizes[key] == 0)
                use_mode[key] = True
            else:
                use_mode[key] = False

        perception_interface = Perceptions(
            batch_size=config_input.perception_batch_sizes,
            gpu_assignment=config_input.perception_gpus,
            compute_methods={},
            viz_methods={},
            num_replicates=config_input.perception_num_replicates,
            path_config=config_input.perception_paths,
            **use_mode
        )
        time.sleep(config_input.perception_initialization_sleep)

    else:
        perception_interface = None

    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.per_process_gpu_memory_fraction = float(memory_fraction)
    sess = tf.Session(config=config_gpu)


    dataset_manager = DatasetManager(conf_module.configInput(), perception_interface)
    batch_tensor = dataset_manager.train.get_batch_tensor()
    batch_tensor_val = dataset_manager.validation.get_batch_tensor()
    dataset_manager.start_training_queueing(sess)
    dataset_manager.start_validation_queueing(sess)

    training_manager = TrainManager(conf_module.configTrain(), None, placeholder_input=False, batch_tensor=batch_tensor)
    if hasattr(conf_module.configTrain(), 'seg_network_erfnet_one_hot'):
        print("Bulding: seg_network_erfnet_one_hot")
        training_manager.build_seg_network_erfnet_one_hot()
    else:
        print("Bulding: standard_network")
        training_manager.build_network()
    training_manager.build_loss()
    training_manager.build_optimization()

    sess.run(tf.global_variables_initializer())

    # leave the segmentation model there, since we want to rerun his model sometime
    if config_main.segmentation_model != None:
        print("the segmentation model name is: ", config_main.segmentation_model_name)
        variables_to_restore = slim.get_variables(
            scope=str(config_main.segmentation_model_name))
        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)
        restore_session(sess, saver, config_main.segmentation_model)

        variables_to_restore = list(set(tf.global_variables()) - set(
            slim.get_variables(scope=str(config_main.segmentation_model_name))))
    else:
        variables_to_restore = tf.global_variables()

    saver = tf.train.Saver(variables_to_restore, max_to_keep=0)
    cpkt = restore_session(sess, saver, config_main.models_path)
    initialIteration = get_last_iteration(cpkt)

    all_saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

    # Creates a manager to manger the screen output and also validation outputs
    if config_main.output_is_on:
        output_manager = OutputManager(conf_module.configOutput(), training_manager, conf_module.configTrain(), sess,
                                       batch_tensor_val)

    # Creates a test manager that connects to a server and tests there constantly

    for i in range(initialIteration, config_main.number_iterations):
        start_time = time.time()
        if i % 3000 == 0:
            if config_main.segmentation_model != None:
                save_model(saver, sess, config_main.models_path + '/ctrl', i)
            save_model(all_saver, sess, config_main.models_path, i)

        #print("running a step")
        training_manager.run_train_step(batch_tensor, sess, i)
        #print("finished a step")

        duration = time.time() - start_time

        #   """ With the current trained net, let the outputmanager print and save all the outputs """
        if config_main.output_is_on:
            output_manager.print_outputs(i, duration)
