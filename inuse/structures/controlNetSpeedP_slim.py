import tensorflow as tf
import numpy as np
from random import randint
from network import Network

def create_structure(tf, x,input_data, input_size,dropout,config):


	
	branches = []

	network_manager = Network(config,dropout)

	"""conv1"""
	x = network_manager.conv_block(x,5,2,16,padding_in='VALID')
	print x


	"""conv2"""
	x = network_manager.conv_block(x,5,2,32,padding_in='VALID')
	print x

	print x
	"""conv3"""
	x = network_manager.conv_block(x,3,2,64,padding_in='VALID')
	print x
	"""conv4"""
	x = network_manager.conv_block(x,3,2,128,padding_in='VALID')
	print x
	"""mp3 (default values)""" 


	""" reshape """

	x =  tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))],name = 'reshape')
	print x

	""" fc1 """

	x = network_manager.fc_block(x,512)
	print x
	""" fc2 """
	x = network_manager.fc_block(x,256)

	"""Process Control"""


	#control = tf.reshape(control, [-1, int(np.prod(control.get_shape()[1:]))],name = 'reshape_control')
	#print control

	""" fc3 """
	with tf.name_scope("Speed"):
		speed = input_data[config.inputs_names.index("Speed")] # get the speed from input data

		speed = network_manager.fc(speed,64)
		speed = network_manager.activation(speed)
		speed = network_manager.fc(speed,64)
		speed = network_manager.activation(speed) 


	"""Start BRANCHING"""


	for i in range(0,len(config.branch_config)):  

		with tf.name_scope("Branch_" + str(i) + "_" + config.branch_config[i][0]):
			if config.branch_config[i][0] == "Gas":
				branch_output=network_manager.fc_block(tf.concat([x,speed],1),256)
			else:
				branch_output =network_manager.fc_block(x,256)
				
			branches.append(network_manager.fc(branch_output,len(config.branch_config[i])))


		

		print branch_output


	""" fc3 """


	weights = network_manager.get_weigths_dict()

	features = network_manager.get_feat_tensors_dict()
	
	



	return branches,None,features,weights
