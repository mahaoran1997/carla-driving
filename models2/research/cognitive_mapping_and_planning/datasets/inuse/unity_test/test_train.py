 
import logging
import sys
sys.path.append('../configuration')
sys.path.append('../input')
sys.path.append('../train')
from config import *
from dataset_manager  import *
from training_manager import TrainManager


def restore_session(sess,saver,models_path):

  ckpt = 0
  if not os.path.exists(models_path):
    os.mkdir( models_path)
  
  ckpt = tf.train.get_checkpoint_state(models_path)
  if ckpt:
    print 'Restoring from ',ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
  else:
    ckpt = 0

  return ckpt



    

def save_model(saver,sess,models_path,i):

  saver.save(sess, models_path + 'model.ckpt', global_step=i)
  print 'Model saved.'


def get_last_iteration(ckpt):

  if ckpt:
    return int(ckpt.model_checkpoint_path.split('-')[1])
  else:
    return 1

    
def test_train(gpu_number):



  """ Initialize the input class to get the configuration """
  config= configMain()


  manager = DatasetManager(config.config_input)



  """ Get the batch tensor that is going to be used around """
  batch_tensor = manager.train.get_batch_tensor()
  batch_tensor_val = manager.validation.get_batch_tensor()
  config_gpu = tf.ConfigProto()
  config_gpu.gpu_options.visible_device_list=gpu_number
  sess = tf.Session(config=config_gpu)
  manager.start_training_queueing(sess)
  manager.start_validation_queueing(sess)




  training_manager= TrainManager(config.config_train)


  training_manager.build_network()

  training_manager.build_loss()

  training_manager.build_optimization()

  """ Initializing Session as variables that control the session """
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.all_variables(),max_to_keep=0)
 


  """Load a previous model if it is configured to restore """
  cpkt = 0
  if config.config_train.restore:
    cpkt = restore_session(sess,saver,config.models_path)


  for i in range(10):  # RUn a few training steps


    training_manager.run_train_step(batch_tensor,sess,i)