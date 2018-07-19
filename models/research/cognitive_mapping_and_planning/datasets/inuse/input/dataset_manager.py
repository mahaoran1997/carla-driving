#import caffe

import numpy as np
from PIL import Image     
import random
import bisect
import os.path

import h5py
import math
import sys

sys.path.append('spliter')

from spliter import Spliter

import tensorflow as tf
import matplotlib.pyplot as plt
import threading
from dataset import *

# TODO: Divide also by acceleration and Steering 


class DatasetManager(object):


  def __init__(self,config):
    print (config.train_db_path)

    """ Read all hdf5_files """
    self._images_train, self._datasets_train = self.read_all_files(config.train_db_path,config.sensor_names,config.dataset_names)

    self._images_val, self._datasets_val = self.read_all_files(config.val_db_path,config.sensor_names, config.dataset_names)


    #print len(self._images_train)
    #print self._images_train[0].shape
    #print self._images_train
    
    spliter = Spliter(1,1,config.steering_bins_perc)  

    #print self._datasets_train[0][config.variable_names.index("Steer")][:]
    divided_keys_train = spliter.divide_keys_by_labels(self._datasets_train[0][config.variable_names.index("Control")][:],config.labels_per_division)

    self._splited_keys_train = spliter.split_by_output(self._datasets_train[0][config.variable_names.index("Steer")][:],divided_keys_train)
    #np.set_printoptions(threshold=np.nan)
    #print self._datasets_train[0][config.variable_names.index("Steer")][:]
    #print np.sort(self._datasets_train[0][config.variable_names.index("Control")][:])

    divided_keys_val = spliter.divide_keys_by_labels(self._datasets_val[0][config.variable_names.index("Control")][:],config.labels_per_division) # THE NOISE IS NOW NONE, TEST THIS


    self._splited_keys_val = spliter.split_by_output( self._datasets_val[0][config.variable_names.index("Steer")][:],divided_keys_val)


    #self._splited_keys_train = self.divide_keys_by_labels(control_train,self._splited_keys_train,config.number_of_divisions,config.labels_per_division)



    

    print max(max(max(self._splited_keys_train)))
    print 'Min ID Train'
    print min(min(min(self._splited_keys_train)))

    print 'Max Id Val'
    print max(max(max(self._splited_keys_val)))
    print len(self._splited_keys_train[0][0]),len(self._splited_keys_train[0][1]),len(self._splited_keys_train[0][2])


    self.train = Dataset(self._splited_keys_train,self._images_train, self._datasets_train,config,config.augment)
    self.validation = Dataset(self._splited_keys_val,self._images_val, self._datasets_val,config,[None]*len(config.sensor_names))  



 
  def start_training_queueing(self,sess):

    enqueue_thread = threading.Thread(target=self.train.enqueue, args=[sess])
    enqueue_thread.isDaemon()
    enqueue_thread.start()

    coord = tf.train.Coordinator()
    self._threads_train = tf.train.start_queue_runners(coord=coord, sess=sess)



  def start_validation_queueing(self,sess):

    enqueue_thread = threading.Thread(target=self.validation.enqueue, args=[sess])
    enqueue_thread.isDaemon()
    enqueue_thread.start()

    coord = tf.train.Coordinator()
    self._threads_val = tf.train.start_queue_runners(coord=coord, sess=sess)


  def stop_training_queueing(self,sess):
    sess.run(self.train.queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(self._threads_train)

  def stop_training_queueing(self,sess):
    sess.run(self.train.queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(self._threads_train)
    


  def read_all_files(self,file_names,image_dataset_names,dataset_names):
   



    datasets_cat = [list([]) for _ in xrange(len(dataset_names))]

    images_data_cat = [list([]) for _ in xrange(len(image_dataset_names))]



    lastidx = 0
    count =0
    #print file_names
    for cword in file_names:
      try:
          #print cword
          #print count
          dset = h5py.File(cword, "r")  

          for i in range(len(image_dataset_names)):
            #print image_dataset_names[i]
            x = dset[image_dataset_names[i]]
            #print x
            old_shape = x.shape[0]
            #print old_shape

            images_data_cat[i].append((lastidx, lastidx+x.shape[0], x))


          for i in range(len(dataset_names)):

            dset_to_append = dset[dataset_names[i]]


            datasets_cat[i].append( dset_to_append[:])



          
          lastidx += old_shape
          dset.flush()
          count +=1

      except IOError:
        import traceback
        exc_type, exc_value, exc_traceback  = sys.exc_info()
        traceback.print_exc()
        traceback.print_tb(exc_traceback,limit=1, file=sys.stdout)
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
        print "failed to open", cword




    for i in range(len(dataset_names)):     
      datasets_cat[i] = np.concatenate(datasets_cat[i], axis=0)
      datasets_cat[i] = datasets_cat[i].transpose((1,0))



    return images_data_cat,datasets_cat


