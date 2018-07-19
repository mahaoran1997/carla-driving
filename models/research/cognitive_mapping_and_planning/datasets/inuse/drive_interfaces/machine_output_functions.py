import numpy as np
import math

from codification import *

from PIL import Image


def input(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  control = np.array(encode(control_input))
  control = control.reshape((1,4))
  speed = speed.reshape((1,1))

  net = branches[0]


  #print clip_input.shape

  #input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))


  #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))
      
      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,input_control:control,dout: [1]*len(config.dropout) }


  output_net = sess.run(net, feed_dict=feedDict)


  
  predicted_steers = (output_net[0][0])

  predicted_acc = (output_net[0][1])


  predicted_brake = (output_net[0][2])
      
  return  predicted_steers,predicted_acc,predicted_brake


def input_drc(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  control = np.array(encode(control_input))
  control = control.reshape((1,4))
  speed = speed.reshape((1,1))

  net = branches[0]


  #print clip_input.shape

  #input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))


  #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))
      
      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,input_control:control,dout: [1]*len(config.dropout) }


  output_net = sess.run(net, feed_dict=feedDict)


  
  predicted_steers = (output_net[0][0])

  predicted_acc = (output_net[0][1])


  return  predicted_steers,predicted_acc,0

def goal(image_input,speed,goal_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_goal=  train_manager._input_data[config.inputs_names.index("Goal")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)
  #aux =goal_input[0] 
  #goal_input[0]  = goal_input[1]
  #goal_input[1] =aux
  module =math.sqrt(goal_input[0] *goal_input[0]  + goal_input[1] *goal_input[1]) 
  goal_input = np.array(goal_input)
  goal_input = goal_input.reshape((1,2))/module
  speed = speed.reshape((1,1))

  net = branches[0]
  print " Inputing ",goal_input


  feedDict = {x: image_input,input_speed:speed,input_goal:goal_input,dout: [1]*len(config.dropout) }


  output_net = sess.run(net, feed_dict=feedDict)


  
  predicted_steers = (output_net[0][0])

  predicted_acc = (output_net[0][1])


  predicted_brake = (output_net[0][2])
      
  return  predicted_steers,predicted_acc,predicted_brake


def base_no_speed(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  net = branches[0]



  feedDict = {x: image_input,dout: [1]*len(config.dropout) }


  output_net = sess.run(net, feed_dict=feedDict)


  
  predicted_steers = (output_net[0][0])

      
  return  predicted_steers,None,None


def base(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)


  speed = speed.reshape((1,1))

  net = branches[0]



  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


  output_net = sess.run(net, feed_dict=feedDict)


  
  predicted_steers = (output_net[0][0])

  predicted_acc = (output_net[0][1])


  predicted_brake = (output_net[0][2])
      
  return  predicted_steers,predicted_acc,predicted_brake


def base_drc(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)


  speed = speed.reshape((1,1))

  net = branches[0]



  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


  output_net = sess.run(net, feed_dict=feedDict)


  
  predicted_steers = (output_net[0][0])

  predicted_acc = (output_net[0][1])
      
  return  predicted_steers,predicted_acc,0

def branched_speed(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==2 or control_input==0.0:
    steer_net = branches[0]
  elif control_input == 3:
    steer_net = branches[2]
  elif control_input == 4:
    steer_net = branches[3]
  elif control_input == 5:
    steer_net = branches[1]

  acc_net = branches[4]
  brake_net = branches[5]
  speed_net = branches[6]  # This is hardcoded !!!!!!

  #print clip_input.shape

  #input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))


  #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))
      
      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


  output_steer = sess.run(steer_net, feed_dict=feedDict)
  output_acc = sess.run(acc_net, feed_dict=feedDict)
  output_speed = sess.run(speed_net, feed_dict=feedDict)
  output_brake = sess.run(brake_net, feed_dict=feedDict)


  if config.use_speed_trick:
    if speed < (4.0/config.speed_factor) and output_speed[0][0] > (4.0/config.speed_factor):  # If (Car Stooped) and ( It should not have stoped)
     output_acc[0][0] =  0.3*(4.0/config.speed_factor -speed) + output_acc[0][0]  #print "DURATION"



  
  predicted_steers = (output_steer[0][0])

  predicted_acc = (output_acc[0][0])


  predicted_brake = (output_brake[0][0])
      
  return  predicted_steers,predicted_acc,predicted_brake




def single_branch_seg(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]
  #image_result = Image.fromarray((image_input*255).astype(np.uint8))
  #image_result.save('image.png')

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))



  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==2 or control_input==0.0:
    all_net = branches[0]
  elif control_input == 3:
    all_net = branches[2]
  elif control_input == 4:
    all_net = branches[3]
  elif control_input == 5:
    all_net = branches[1]


      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


  output_all = sess.run(all_net, feed_dict=feedDict)
  
  predicted_steers = (output_all[0][0])

  predicted_acc = (output_all[0][1])

  predicted_brake = (output_all[0][2])

  predicted_speed =  sess.run(branches[4], feed_dict=feedDict)
  predicted_speed = predicted_speed[0][0]
  real_speed = speed*config.speed_factor
  print ' REAL PREDICTED ',predicted_speed*config.speed_factor
  print ' acc ',predicted_acc
  print ' REAL SPEED ',real_speed
  real_predicted =predicted_speed*config.speed_factor
  if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
    print 'BOOSTING'
    predicted_acc =  1*(20.0/config.speed_factor -speed) + predicted_acc  #print "DURATION"

    predicted_brake=0.0
    predicted_acc = predicted_acc[0][0]

  print predicted_steers,predicted_acc,predicted_brake

  return  predicted_steers,predicted_acc,predicted_brake


def single_branch(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]
  #image_result = Image.fromarray((image_input*255).astype(np.uint8))
  #image_result.save('image.png')

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==2 or control_input==0.0:
    all_net = branches[0]
  elif control_input == 3:
    all_net = branches[2]
  elif control_input == 4:
    all_net = branches[3]
  elif control_input == 5:
    all_net = branches[1]


  #print clip_input.shape

  #input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

  #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))


      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


  output_all = sess.run(all_net, feed_dict=feedDict)


  predicted_steers = (output_all[0][0])

  predicted_acc = (output_all[0][1])

  predicted_brake = (output_all[0][2])

  predicted_speed =  sess.run(branches[4], feed_dict=feedDict)
  predicted_speed = predicted_speed[0][0]
  real_speed = speed*config.speed_factor
  print ' REAL PREDICTED ',predicted_speed*config.speed_factor

  print ' REAL SPEED ',real_speed
  real_predicted =predicted_speed*config.speed_factor
  if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
    print 'BOOSTING'
    predicted_acc =  1*(20.0/config.speed_factor -speed) + predicted_acc  #print "DURATION"

    predicted_brake=0.0

    predicted_acc = predicted_acc[0][0]
    
  return  predicted_steers,predicted_acc,predicted_brake



def single_branch_wp(image_input,speed,control_input,config,sess,train_manager):
  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==2 or control_input==0.0:
    all_net = branches[0]
  elif control_input == 3:
    all_net = branches[2]
  elif control_input == 4:
    all_net = branches[3]
  elif control_input == 5:
    all_net = branches[1]


  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }

  output_all = sess.run(all_net, feed_dict=feedDict)

  predicted_wp1_angle = (output_all[0][0])

  predicted_wp2_angle = (output_all[0][1])

  predicted_steers = (output_all[0][2])

  predicted_acc = (output_all[0][3])

  predicted_brake = (output_all[0][4])

  predicted_speed =  sess.run(branches[4], feed_dict=feedDict)
  predicted_speed = predicted_speed[0][0]
  real_speed = speed*config.speed_factor
  print ' REAL PREDICTED ',predicted_speed*config.speed_factor

  print ' REAL SPEED ',real_speed
  real_predicted =predicted_speed*config.speed_factor
  if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
    print 'BOOSTING'
    predicted_acc =  1*(20.0/config.speed_factor -speed) + predicted_acc  #print "DURATION"

    predicted_brake=0.0

    predicted_acc = predicted_acc[0][0]


  return  predicted_steers,predicted_acc,predicted_brake,predicted_wp1_angle,predicted_wp2_angle



def single_branch_drc(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==2 or control_input==0.0:
    all_net = branches[0]
  elif control_input == 3:
    all_net = branches[2]
  elif control_input == 4:
    all_net = branches[3]
  elif control_input == 5:
    all_net = branches[1]


  #print clip_input.shape

  #input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))


  #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))
      
      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }

  output_all = sess.run(all_net, feed_dict=feedDict)


  predicted_steers = (output_all[0][0])

  predicted_acc = (output_all[0][1])


  return  predicted_steers,predicted_acc,0


def branched_speed_4cmd(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==5:
    steer_net = branches[1]
  elif control_input == 3 or control_input == 6:
    steer_net = branches[2]
  elif control_input == 4 or control_input == 7 or control_input == 8:
    steer_net = branches[3]
  else:
    steer_net = branches[0]

  acc_net = branches[4]
  brake_net = branches[5]
  speed_net = branches[6]  # This is hardcoded !!!!!!

  #print clip_input.shape

  #input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))


  #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))
      
      
  #image_result.save(str('saida_res.jpg'))

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


  output_steer = sess.run(steer_net, feed_dict=feedDict)
  output_acc = sess.run(acc_net, feed_dict=feedDict)
  output_speed = sess.run(speed_net, feed_dict=feedDict)
  output_brake = sess.run(brake_net, feed_dict=feedDict)


  if config.use_speed_trick:
    if speed < (4.0/config.speed_factor) and output_speed[0][0] > (4.0/config.speed_factor):  # If (Car Stooped) and ( It should not have stoped)
     output_acc[0][0] =  0.3*(4.0/config.speed_factor -speed) + output_acc[0][0]  #print "DURATION"



  
  predicted_steers = (output_steer[0][0])

  predicted_acc = (output_acc[0][0])


  predicted_brake = (output_brake[0][0])
      
  return  predicted_steers,predicted_acc,predicted_brake


def get_intermediate_rep(image_input,speed,config,sess,train_manager):

  seg_network = train_manager._gray
  x = train_manager._input_images 
  dout = train_manager._dout
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]

  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))
  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }

  output_image = sess.run(seg_network, feed_dict=feedDict)

  return output_image[0]



def vbp(image_input,speed,config,sess,train_manager):

  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  vbp_images_tensor = train_manager._vis_images

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }  

  vbp_images = sess.run(vbp_images_tensor, feed_dict=feedDict)

  return vbp_images[0]


def seg_viz (image_input,speed,config,sess,train_manager):

  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

  #print ('Image Size Tensor: ', 1,config.image_size[0],config.image_size[1],config.image_size[2])

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  gray_images_tensor = train_manager._gray

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }  

  gray_images = sess.run(gray_images_tensor, feed_dict=feedDict)

  return gray_images


def vbp_nospeed(image_input,config,sess,train_manager):

  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))


  vbp_images_tensor = train_manager._vis_images

  feedDict = {x: image_input,dout: [1]*len(config.dropout) }  

  vbp_images = sess.run(vbp_images_tensor, feed_dict=feedDict)


  return vbp_images[0]
