import logging
import sys
sys.path.append('configuration')
sys.path.append('input')
sys.path.append('train')
sys.path.append('output')
#from config import *
from dataset_manager  import *
from training_manager import TrainManager
from training_manager import restore_session
from training_manager import get_last_iteration
from training_manager import save_model

from output_manager import OutputManager
from test_manager import TestManager

slim = tf.contrib.slim
from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables


def train(gpu_number, experiment_name,path,memory_fraction,port):



  """ Initialize the input class to get the configuration """

  conf_module = __import__(experiment_name)

  config = conf_module.configMain()

  config_gpu = tf.ConfigProto()
  config_gpu.gpu_options.per_process_gpu_memory_fraction=float(memory_fraction)
  sess = tf.Session(config=config_gpu)
  

  manager = DatasetManager(conf_module.configInput(path))

  """ Get the batch tensor that is going to be used around """
  batch_tensor = manager.train.get_batch_tensor()
  batch_tensor_val = manager.validation.get_batch_tensor()


  manager.start_training_queueing(sess)
  manager.start_validation_queueing(sess)


  training_manager= TrainManager(conf_module.configTrain(),None)


  if hasattr(conf_module.configTrain(), 'rgb_seg_network_one_hot'):
    print("Bulding: rgb_seg_network_one_hot")
    training_manager.build_rgb_seg_network_one_hot()
  
  else:
    if hasattr(conf_module.configTrain(), 'seg_network_gt_one_hot'):
      print("Bulding: seg_network_gt_one_hot")
      training_manager.build_seg_network_gt_one_hot()

    else:
      if hasattr(conf_module.configTrain(), 'seg_network_gt_one_hot_join'):
        print("Bulding: seg_network_gt_one_hot_join")
        training_manager.build_seg_network_gt_one_hot_join()

      else:
        if hasattr(conf_module.configTrain(), 'rgb_seg_network_enet'):
          print("Bulding: rgb_seg_network_enet")
          training_manager.build_rgb_seg_network_enet()

        else:
          if hasattr(conf_module.configTrain(), 'rgb_seg_network_enet_one_hot'):
            print("Bulding: rgb_seg_network_enet_one_hot")
            training_manager.build_rgb_seg_network_enet_one_hot()

          else:
            if hasattr(conf_module.configTrain(), 'seg_network_enet_one_hot'):
              print("Bulding: seg_network_enet_one_hot")
              training_manager.build_seg_network_enet_one_hot()

            else:
              if hasattr(conf_module.configTrain(), 'seg_network_erfnet_one_hot'):
                print("Bulding: seg_network_erfnet_one_hot")
                training_manager.build_seg_network_erfnet_one_hot()          

              else:
                print("Bulding: standard_network")
                training_manager.build_network()



  training_manager.build_loss()

  training_manager.build_optimization()


  sess.run(tf.global_variables_initializer())

  if  config.segmentation_model != None:
    exclude = ['global_step']
    print (config.segmentation_model_name)
    variables_to_restore = slim.get_variables(scope=str(config.segmentation_model_name))#config.segmentation_model_name

    saver = tf.train.Saver(variables_to_restore,max_to_keep=0)

    seg_ckpt = restore_session(sess,saver,config.segmentation_model)

    initialIteration = 0
    variables_to_restore = list(set(tf.global_variables()) - set(slim.get_variables(scope=str(config.segmentation_model_name))))#config.segmentation_model_name
  else:
    variables_to_restore = tf.global_variables()

  saver = tf.train.Saver(variables_to_restore,max_to_keep=0)
  cpkt = restore_session(sess,saver,config.models_path)
  initialIteration = get_last_iteration(cpkt)

  all_saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)




  # """Training"""


  
  # Creates a manager to manager the screen output and also validation outputs
  if config.output_is_on:
    output_manager = OutputManager(conf_module.configOutput(),training_manager,conf_module.configTrain(),sess,batch_tensor_val)

  # Creates a test manager that connects to a server and tests there constantly
  if config.perform_simulation_test:

    test_manager = TestManager(conf_module.configTest(),conf_module.configInput(),sess,training_manager,port,experiment_name)


    
  for i in range(initialIteration, config.number_iterations):

    #   """ Get the training batch """
    

    #   start_time = time.time()
    #   """Save the model every 300 iterations"""
    start_time = time.time()
    if i%10000 == 0:

      if  config.segmentation_model != None:
        save_model(saver,sess,config.models_path + '/ctrl',i)

      save_model(all_saver,sess,config.models_path,i)

  

    #   """ Run the training step and monitor its execution time """

    #print ' RUN STEP'
    training_manager.run_train_step(batch_tensor,sess,i)

    #print ' run step done '


    duration = time.time() - start_time

    #   """ With the current trained net, let the outputmanager print and save all the outputs """
    if config.output_is_on:
      output_manager.print_outputs(i,duration) 

    if config.perform_simulation_test and i%config.test_interval ==0:

      test_manager.perform_simulation_test(i)
