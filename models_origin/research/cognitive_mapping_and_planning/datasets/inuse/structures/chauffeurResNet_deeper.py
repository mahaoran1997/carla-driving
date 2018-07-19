import tensorflow as tf
import numpy as np
from random import randint
from network import Network

def create_structure(tf, input_image,input_data, input_size,dropout,config):


	
	branches = []

	x = input_image

	network_manager = Network(config,dropout,tf.shape(x))

	

	"""conv1"""
	xc = network_manager.conv_block(x,5,2,32,padding_in='SAME')

	"""res1"""
	xc = network_manager.residual_block(xc,2,64,padding_in='SAME')

	"""res2"""
	xc = network_manager.residual_block(xc,2,128,padding_in='SAME')

	"""res3"""
	xc = network_manager.residual_block(xc,1,256,padding_in='SAME')

	""" reshape """
	x =  tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))],name = 'reshape')
	print x

	""" fc1 """
	x = network_manager.fc_block(x,512)
	print x
	""" fc2 """
	x = network_manager.fc_block(x,512)



	"""Process Control"""
	#control = tf.reshape(control, [-1, int(np.prod(control.get_shape()[1:]))],name = 'reshape_control')
	#print control

	""" Speed (measurements)"""
	with tf.name_scope("Speed"):
		speed = input_data[config.inputs_names.index("Speed")] # get the speed from input data
		speed = network_manager.fc_block(speed,128)
		speed = network_manager.fc_block(speed,128)

	""" Joint sensory """
	j = tf.concat([x,speed],1)
	j = network_manager.fc_block(j, 512)

	"""Start BRANCHING"""
	for i in range(0,len(config.branch_config)):  
		with tf.name_scope("Branch_" + str(i)):
			if config.branch_config[i][0] == "Speed":
				# we only use the image as input to speed prediction
				branch_output =network_manager.fc_block(x,256)
				branch_output =network_manager.fc_block(branch_output,256)
			else:
				branch_output =network_manager.fc_block(j,256)
				branch_output =network_manager.fc_block(branch_output,256)
				
			branches.append(network_manager.fc(branch_output,len(config.branch_config[i])))


		

		print branch_output


	weights = network_manager.get_weigths_dict()

	features = network_manager.get_feat_tensors_dict()
	vis_images = input_image[:,:,:,0:1]
	#vis_images = network_manager.get_vbp_images(xc)
	print vis_images

	#print vis_images.get_shape(tf.slice(x))

	#vis_images = tf.div(vis_images  -tf.reduce_min(vis_images),tf.reduce_max(vis_images) -tf.reduce_min(vis_images))

	return branches,vis_images,features,weights
