import tensorflow as tf
import network

def mse_input(network_outputs,ground_truths,control_input,config):



	branches_configuration = config.branch_config # Get the branched configuration

	#ground_truth_split = tf.split(ground_truths,ground_truths.get_shape()[1],1 )

	error_vec = []

	energy_vec =[]

	print network_outputs
	for i in range(len(branches_configuration)):

		energy_branch =[]
		error_branch =[]

		#if network_outputs[i].get_shape()[1] == 1: # Size 1 cannot be splited
		#	network_outputs_split = network_outputs[i]
		#else:
		print network_outputs[i]
		network_outputs_split =tf.split(network_outputs[i],network_outputs[i].get_shape()[1],1 )


		print branches_configuration[i]
		for j in range(len(branches_configuration[i])):

			print 'split'
			print network_outputs_split[j]

			# Get the name of the target data to be taken
			target_name = branches_configuration[i][j] # name of the target
			# Get the target gt ( TODO: This should be a dictionary, I think it is more logical)
			print target_name



			target_gt = ground_truths[config.targets_names.index(target_name)]
			if i < config.inputs_sizes[config.inputs_names.index('Control')]:
				branch_selection = tf.reshape(control_input[:,i],tf.shape(ground_truths[0]))
			else:
				branch_selection =tf.ones(tf.shape(target_gt))
			print target_gt
			squared_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[j]),2)
			dist =tf.abs(tf.subtract(target_gt,  network_outputs_split[j])) *branch_selection
			error_branch.append(dist)
			print 'dist'
			print dist



			variable_loss =squared_dist * branch_selection


			print 'loss'
			print variable_loss
			energy_branch.append(variable_loss)
			if i==0 and j==0:
				loss_function = variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name]
			else:
				loss_function = tf.add(variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name],loss_function)


		energy_vec.append(energy_branch)
		error_vec.append(error_branch)

	#exit()

	return loss_function,error_vec,energy_vec,None,branch_selection




def mse_branched(network_outputs,ground_truths,control_input,config):


	branches_configuration = config.branch_config # Get the branched configuration

	#ground_truth_split = tf.split(ground_truths,ground_truths.get_shape()[1],1 )

	error_vec = []

	energy_vec =[]

	print network_outputs
	for i in range(len(branches_configuration)):

		energy_branch =[]
		error_branch =[]

		#if network_outputs[i].get_shape()[1] == 1: # Size 1 cannot be splited
		#	network_outputs_split = network_outputs[i]
		#else:
		print network_outputs[i]
		network_outputs_split =tf.split(network_outputs[i],network_outputs[i].get_shape()[1],1 )


		print branches_configuration[i]
		for j in range(len(branches_configuration[i])):

			print 'split'
			print network_outputs_split[j]

			# Get the name of the target data to be taken
			target_name = branches_configuration[i][j] # name of the target
			# Get the target gt ( TODO: This should be a dictionary, I think it is more logical)
			print target_name


			target_gt = ground_truths[config.targets_names.index(target_name)]
			if i < config.inputs_sizes[config.inputs_names.index('Control')]:
				branch_selection = tf.reshape(control_input[:,i],tf.shape(ground_truths[0]))
			else:
				branch_selection =tf.ones(tf.shape(target_gt))
			print target_gt
			squared_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[j]),2)
			dist =tf.abs(tf.subtract(target_gt,  network_outputs_split[j])) * branch_selection
			error_branch.append(dist)
			print 'dist'
			print dist


			variable_loss =squared_dist * branch_selection


			print 'loss'
			print variable_loss
			energy_branch.append(variable_loss)
			if i==0 and j==0:
				loss_function = variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name]
			else:
				loss_function = tf.add(variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name],loss_function)
				
		energy_vec.append(energy_branch)
		error_vec.append(error_branch)

	#exit()

	return loss_function,error_vec,energy_vec,None,branch_selection


def mse_branched_variational_weights(network_outputs,ground_truths,control_input,config):



	branches_configuration = config.branch_config # Get the branched configuration


	variance = tf.get_variable(name='variance', shape=[120,1],initializer=tf.contrib.layers.xavier_initializer())
   

	error_vec = []

	energy_vec =[]

	print network_outputs
	for i in range(len(branches_configuration)):

		energy_branch =[]
		error_branch =[]

		#if network_outputs[i].get_shape()[1] == 1: # Size 1 cannot be splited
		#	network_outputs_split = network_outputs[i]
		#else:
		print network_outputs[i]
		network_outputs_split =tf.split(network_outputs[i],network_outputs[i].get_shape()[1],1 )


		print branches_configuration[i]
		for j in range(len(branches_configuration[i])):

			print 'split'
			print network_outputs_split[j]

			# Get the name of the target data to be taken
			target_name = branches_configuration[i][j] # name of the target
			# Get the target gt ( TODO: This should be a dictionary, I think it is more logical)
			print target_name



			target_gt = ground_truths[config.targets_names.index(target_name)]
			if i < config.inputs_sizes[config.inputs_names.index('Control')]:
				branch_selection = tf.reshape(control_input[:,i],tf.shape(ground_truths[0]))
			else:
				branch_selection =tf.ones(tf.shape(target_gt))
			print target_gt
			squared_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[j]),2)
			dist =tf.abs(tf.subtract(target_gt,  network_outputs_split[j])) * branch_selection
			error_branch.append(dist)
			print 'dist'
			print dist




			variable_loss = squared_dist * branch_selection
			variable_loss = (1/(2*tf.exp(variance*variance)) )*variable_loss + variance*variance

			print 'loss'
			print variable_loss
			energy_branch.append(variable_loss)
			if i==0 and j==0:
				loss_function = variable_loss
			else:
				loss_function = tf.add(variable_loss,loss_function)
				
		energy_vec.append(energy_branch)
		error_vec.append(error_branch)

	#exit()

	return loss_function,error_vec,energy_vec,None,branch_selection




def mse_seg_branched(network_outputs,seg_output,ground_truths,image_seg_ground_truth,control_input,config):


	onehot_labels = network.image_to_one_hot(image_seg_ground_truth,config.number_of_labels)


	class_weights = [47.075315159067941, 14.802265924071287, 3.3533567319644777, 6.2486706964225798, 2.5176]


	weights = onehot_labels * class_weights
   	weights = tf.reduce_sum(weights, 3)
   	image_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=seg_output, weights=weights)


	branches_configuration = config.branch_config # Get the branched configuration

	#ground_truth_split = tf.split(ground_truths,ground_truths.get_shape()[1],1 )

	error_vec = []

	energy_vec =[]

	print network_outputs
	for i in range(len(branches_configuration)):

		energy_branch =[]
		error_branch =[]

		#if network_outputs[i].get_shape()[1] == 1: # Size 1 cannot be splited
		#	network_outputs_split = network_outputs[i]
		#else:
		print network_outputs[i]
		network_outputs_split =tf.split(network_outputs[i],network_outputs[i].get_shape()[1],1 )


		print branches_configuration[i]
		for j in range(len(branches_configuration[i])):

			print 'split'
			print network_outputs_split[j]

			# Get the name of the target data to be taken
			target_name = branches_configuration[i][j] # name of the target
			# Get the target gt ( TODO: This should be a dictionary, I think it is more logical)
			print target_name



			target_gt = ground_truths[config.targets_names.index(target_name)]
			if i < config.inputs_sizes[config.inputs_names.index('Control')]:
				branch_selection = tf.reshape(control_input[:,i],tf.shape(ground_truths[0]))
			else:
				branch_selection =tf.ones(tf.shape(target_gt))
			print target_gt
			squared_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[j]),2)
			dist =tf.abs(tf.subtract(target_gt,  network_outputs_split[j])) * branch_selection
			error_branch.append(dist)
			print 'dist'
			print dist




			variable_loss =squared_dist * branch_selection


			print 'loss'
			print variable_loss
			energy_branch.append(variable_loss)
			if i==0 and j==0:
				loss_function = variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name] + image_loss
				loss_control =variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name]
			else:
				loss_function = tf.add(variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name],loss_function)	
				loss_control = tf.add(variable_loss*config.branch_loss_weight[i]*config.variable_weight[target_name],loss_control)
				
		energy_vec.append(energy_branch)
		error_vec.append(error_branch)

	#exit()

	return loss_function,error_vec,energy_vec,loss_control,branch_selection

