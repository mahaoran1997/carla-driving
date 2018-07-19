
import numpy as np

from PIL import Image     
import random
import bisect
import os.path
import h5py
import traceback
import time
import math
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from codification import *
import cv2


def join_classes(labels_image,labels_mapping):

  compressed_labels_image = np.copy(labels_image) 

  for key,value in labels_mapping.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value

  return compressed_labels_image


class Dataset(object):




  def __init__(self,splited_keys,images,datasets ,config_input,augmenter):


    self._splited_keys = splited_keys
    self._images = images
    self._variables = np.concatenate(tuple(datasets), axis=0)  # Cat the datasets

    self._positions_to_train = range(0,config_input.number_steering_bins) # WARNING THIS NEED TO BE A MULTIPLE OF THE NUMBER OF CLIPS

    self._iteration =0


    self._augmenter = augmenter

    self._config = config_input
    self._batch_size = config_input.batch_size



    self._queue_image_input = tf.placeholder(tf.float32, shape=[config_input.batch_size, config_input.image_size[0],config_input.image_size[1],config_input.image_size[2]])

    self._queue_targets = []
    self._queue_inputs = []
    self._queue_shapes =[[config_input.image_size[0],config_input.image_size[1],config_input.image_size[2]]]


    for i in range(len(self._config.targets_names)):
      self._queue_targets.append(tf.placeholder(tf.float32, shape=[config_input.batch_size, self._config.targets_sizes[i]]))
      self._queue_shapes.append([self._config.targets_sizes[i]])

    for i in range(len(self._config.inputs_names)):
      self._queue_inputs.append(tf.placeholder(tf.float32, shape=[config_input.batch_size, self._config.inputs_sizes[i]]))
      self._queue_shapes.append([self._config.inputs_sizes[i]])


    print [tf.float32]*(len(self._config.targets_names)+len(self._config.inputs_names))

    
    print [self._queue_image_input] + self._queue_targets + self._queue_inputs 

    self._queue = tf.FIFOQueue(capacity=config_input.queue_capacity, dtypes=[tf.float32]*(len(self._config.targets_names)+len(self._config.inputs_names) + 1), \
                      shapes= self._queue_shapes)



    self._enqueue_op = self._queue.enqueue_many([self._queue_image_input] + self._queue_targets + self._queue_inputs )
    self._dequeue_op = self._queue.dequeue()

    self._batch_tensor = tf.train.batch(self._dequeue_op, batch_size=config_input.batch_size, capacity=config_input.queue_capacity)



  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels

  def get_batch_tensor(self):
    return self._batch_tensor


  def get_data_by_ids(self,generated_ids,batch_size):

    X_batch = np.zeros((batch_size, 1, self._input_size[0], self._input_size[1],self._input_size[2]), dtype='uint8')
    count = 0
    #print generated_ids
    for i in generated_ids:
      i=int(i)
      for es, ee, x in self._images:
        #print es
        #print i
        #print ee
        #print x.shape
        if i >= es and i < ee:
        #print x[]

          image = np.array(x[i-es-1+1:i-es+1,:,:,:])
          #print image

          X_batch[count] = image
          break


      count += 1
    return X_batch





  def sample_positions_to_train(self,number_of_samples):
    def sample_from_vec(vector):


      sample_number = random.choice(vector)

      # Remove the sampled position from the main list and the splited_list
      del self._positions_to_train[self._positions_to_train.index(sample_number)]
      del vector[vector.index(sample_number)]
      # Refil if is the case
     

      return sample_number,vector

    # Divide it into 3 equal parts
    sample_positions = []
    splited_list =[]
    if len(self._positions_to_train) ==0:
      self._positions_to_train=range(0,self._config.number_steering_bins)

    for i in range(0,3):
      position = i*(len(self._positions_to_train)/3)
      if i==2:
        splited_list.append(self._positions_to_train[position:])
      else:
        splited_list.append(self._positions_to_train[position:position + len(self._positions_to_train)/3])
    #splited_list[2].append(splited_list[3][0])
    #del splited_list[3]  
    #print splited_list

    sample_id = 0
    #print number_of_samples
    #print "Positions to Train"
    #print len(self._positions_to_train)
    while sample_id < number_of_samples:
      # Sample Mid
      if len(splited_list[1]) > 0 :
      
        sampled_value,splited_list[1] = sample_from_vec(splited_list[1])
        #print sampled_value,splited_list[1]

        sample_positions.append(sampled_value)
        sample_id+=1
        if sample_id >= number_of_samples:
          break

        if len(self._positions_to_train) ==0:
          self._positions_to_train = range(0,self._number_steering_levels)
          splited_list =[]
          for i in range(0,3):
            position = i*(len(self._positions_to_train)/3)
            if i==2:
              splited_list.append(self._positions_to_train[position:])
            else:
              splited_list.append(self._positions_to_train[position:position + len(self._positions_to_train)/3])
        
        
      # Sample Left

      if len(splited_list[0]) > 0:
      
        sampled_value,splited_list[0] = sample_from_vec(splited_list[0])
        sample_positions.append(sampled_value)
        sample_id+=1
        if sample_id >= number_of_samples:
          break
        
        if len(self._positions_to_train) ==0:
          self._positions_to_train = range(0,self._config.number_steering_bins)
          splited_list =[]
          for i in range(0,3):
            position = i*(len(self._positions_to_train)/3)
            if i==2:
              splited_list.append(self._positions_to_train[position:])
            else:
              splited_list.append(self._positions_to_train[position:position + len(self._positions_to_train)/3])

        
      # Sample Right
      if len(splited_list[2]) > 0:
      
        sampled_value,splited_list[2] = sample_from_vec(splited_list[2])
        sample_positions.append(sampled_value)
        sample_id+=1
        if sample_id >= number_of_samples:
          break

        if len(self._positions_to_train) ==0:
          self._positions_to_train = range(0,self._config.number_steering_bins)
          splited_list =[]
          for i in range(0,3):
            position = i*(len(self._positions_to_train)/3)
            if i==2:
              splited_list.append(self._positions_to_train[position:])
            else:
              splited_list.append(self._positions_to_train[position:position + len(self._positions_to_train)/3])

        
        

    return sample_positions


    


  def datagen(self,time_len, batch_size,number_control_divisions):


    sensors_batch = []
    for i in range(len(self._images)):
      sensors_batch.append( np.zeros((batch_size, self._config.sensors_size[i][0],\
       self._config.sensors_size[i][1],self._config.sensors_size[i][2]), dtype='uint8'))
    generated_ids = np.zeros((batch_size),dtype='int32')



    while True:
      try:
        t = time.time()
        
        count =0
        start = time.time()
        for control_part in range(0,number_control_divisions):

        

         

          sampled_positions = self.sample_positions_to_train(int(batch_size/3))



          for outer_n in sampled_positions:

            #print ' bin of ',control_part,' has len ', len(self._splited_keys[control_part][outer_n])
            i = random.choice(self._splited_keys[control_part][outer_n]) 


            for s in range(len(self._images)):
              for es, ee, x in self._images[s]:

                if i >= es and i < ee:


                  image = np.array(x[i-es-time_len+1:i-es+1,:,:,:])

                  sensors_batch[s][count,:,:,:] = image
                  break


            generated_ids[count] = i
            count +=1



        return sensors_batch,generated_ids
      except KeyboardInterrupt:
        raise
      except:
        traceback.print_exc()
        pass




  """Return the next `batch_size` examples from this data set."""
  def next_batch(self):
    
    
    batch_size = self._batch_size


    sensors,generated_ids = self.datagen(1 , batch_size,len(self._splited_keys))

    if 'labels' in self._config.sensor_names:
        label_idx = self._config.sensor_names.index('labels')
    else:
        label_idx = -1

    # Get the images -- Perform Augmentation!!!
    for i in range(len(sensors)):
      sensors[i] =  np.array((sensors[i]))

      if i==label_idx: ##labels
        if hasattr(self._config, 'labels_mapping'):
          #print('Joining classes: ', self._config.labels_mapping)
          sensors[i] = join_classes(sensors[i],self._config.labels_mapping)

        if self._augmenter[i] != None:
          sensors[i][np.where(sensors[i]==0)] =6
          sensors[i] = self._augmenter[i].augment_images(sensors[i])
          sensors[i][np.where(sensors[i]==0)] =2*(255/4)
          sensors[i][np.where(sensors[i]==6)] =0

      elif self._augmenter[i] != None:
        sensors[i] = self._augmenter[i].augment_images(sensors[i])


      sensors[i]  = sensors[i].astype(np.float32)


    # Get the targets
    float_data = self._variables[:,generated_ids]
    targets =[]
    for i in range(len(self._config.targets_names)):
      targets.append(np.zeros((batch_size,self._config.targets_sizes[i])))

    # Get the inputs
    inputs = []
    for i in range(len(self._config.inputs_names)):
      inputs.append(np.zeros((batch_size,self._config.inputs_sizes[i])))


    for i in range(0,batch_size):

        for j in range(len(self._images)):
          if self._config.sensors_normalize[j]:
            sensors[j][i,:,:,:] = np.multiply(sensors[j][i,:,:,:],1.0 / 255.0)
          else:
            sensors[j][i,:,:,:] = sensors[j][i,:,:,:]

          


        count =0
        for j in range(len(self._config.targets_names)):
          k = self._config.variable_names.index(self._config.targets_names[j])
          targets[count][i] = float_data[k,i]

          if self._config.targets_names[j] == "Speed":
            targets[count][i]/=self._config.speed_factor

          if self._config.targets_names[j] == "Gas":

            targets[count][i]=max(0,targets[count][i])

          if self._config.targets_names[j] == "Brake":
            targets[count][i]=min(1.0,max(0,targets[count][i]))


          if hasattr(self._config, 'extra_augment_factor') and self._config.targets_names[j] == "Steer" :
            camera_pos = self._config.variable_names.index('Camera')
            speed_pos = self._config.variable_names.index('Speed')

            
            #angle = self._config.variable_names.index('Angle') == 15
            angle = float_data[self._config.variable_names.index('Angle'),i]

            #########Augmentation!!!!
            time_use =  1.0
            car_lenght = 6.0
            speed = math.fabs(float_data[speed_pos,i])
            if angle > 0.0:
              angle = math.radians(math.fabs(angle))
              targets[count][i] -=min(self._config.extra_augment_factor*(math.atan((angle*car_lenght)/(time_use*speed+0.05)))/3.1415,0.3)
            else:
              angle = math.radians(math.fabs(angle))
              targets[count][i]+=min(self._config.extra_augment_factor*(math.atan((angle*car_lenght)/(time_use*speed+0.05)))/3.1415,0.3)
            

            

          if hasattr(self._config, 'saturated_factor') and self._config.targets_names[j] == "Steer" :
            angle = float_data[self._config.variable_names.index('Angle'),i]
            #print angle
            #########Augmentation!!!!
            if angle < 0.0:

              targets[count][i] =1.0
            elif angle >0.0:
              targets[count][i] =-1.0

          count += 1


        count =0
        for j in range(len(self._config.inputs_names)):
          k = self._config.variable_names.index(self._config.inputs_names[j])

          if self._config.inputs_names[j] == "Control":
         
            inputs[count][i] = encode(float_data[k,i])


          if self._config.inputs_names[j] == "Speed":
            inputs[count][i] = float_data[k,i]/self._config.speed_factor

          if self._config.inputs_names[j] == "Distance":
            inputs[count][i] = check_distance(float_data[k,i])


          if self._config.inputs_names[j] == "Goal":
            module = math.sqrt(float_data[k,i]*float_data[k,i] + float_data[k+1,i]*float_data[k+1,i])
            #print 'k ',k
            #print 'module',module
            #print 'float_data',float_data[k,i],float_data[k+1,i]
            float_data[k,i] = float_data[k,i]/module
            float_data[k+1,i] = float_data[k+1,i]/module

            inputs[count][i] = float_data[k:k+2,i]




          count += 1

       

        

    #print targets
    try: # We have both labels and rgb
      labels_pos=self._config.sensor_names.index('labels')
      rgb_pos=self._config.sensor_names.index('rgb')

      fused_rgb_labels = np.concatenate((sensors[rgb_pos],sensors[labels_pos]),axis=3)
      #print 'CONCATENATE'
      return fused_rgb_labels, targets, inputs
      
    except:

      try: # We have labels
        labels_pos=self._config.sensor_names.index('labels')
        return sensors[labels_pos], targets,inputs
      except: # We have RGB
        return sensors[self._config.sensor_names.index('rgb')], targets,inputs
    

  def process_run(self,sess,data_loaded):

    queue_feed_dict={self._queue_image_input:data_loaded[0]} # images we already put by default

    for i in range(len(self._config.targets_names)):

      queue_feed_dict.update({self._queue_targets[i]:data_loaded[1][i]})

    for i in range(len(self._config.inputs_names)):

      queue_feed_dict.update({self._queue_inputs[i]:data_loaded[2][i]})


    #print queue_feed_dict
    sess.run(self._enqueue_op, feed_dict=queue_feed_dict)
    


  def enqueue(self,sess):

    while True:
      #print("starting to write into queue")
      queue_time = time.time()
      


      data_loaded = self.next_batch()


      self.process_run(sess,data_loaded)


