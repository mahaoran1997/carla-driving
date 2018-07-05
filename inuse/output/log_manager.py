"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import tensorflow as tf
from random import *
import sys

sys.path.append('../utils')

from skimage.transform import resize
import terminal_draw_tool
from codification import *
#from image_write import *


class LogManager(object):

	def __init__(self,config,training_manager,sess,curses=False):
		""" If we want to visualize the outputs """
		self._curses = curses
		if curses:
			terminal_draw_tool.start_curses()
		#terminal_draw_tool.start_curses()
		#terminal_draw_tool.curses_started = True

		self._batch_size = config.batch_size 
		self._training_manager = training_manager
		self._sess = sess # the session from tensorflow that is going to be used
		self._config = config
		self._training_start_time =  time.time()



		""" DRAWING STUFF """


		self._lowest_error = 100000
		self._lowest_val_error = 100000
		self._lowest_val_iter =100000
		self._last_val_error = 0
		self._last_train_error =0 
		self._lowest_iter =0 
		self._epoch_number = 0
		self._average_val_error = 0.0
		self._change_val = 0
		self._current_velocity = 0
		self.split_target = []
		for i in self._config.targets_to_print:

			output_tensor = self._training_manager._output_network[self._config.selected_branch] 
			self.split_target.append(tf.split(output_tensor,output_tensor.get_shape()[1],1))
			


		self._last_batch_error= []
		self._last_batch_targets = []
		self._last_batch_inputs = []
		self._last_batch_energy = []
		self._last_batch_gts = []
		#self._last_batch_gain =[]
		#self._last_prob_dist = []
		#self._last_energy_contrib_vec =[]  # The latest energy contribution



	# TODO: to be repaired in order to remove the unnecessary opencv dependance.

	"""
	def write_images(self):
		feedDict = self._training_manager.get_feed_dict()
		images =feedDict[self._training_manager._input_images]
		steer =feedDict[self._training_manager._targets_data[self._config.targets_names.index("Steer")]]
		gas =feedDict[self._training_manager._targets_data[self._config.targets_names.index("Gas")]]
		input_control =feedDict[self._training_manager._input_data[self._config.inputs_names.index("Control")]]
		id_image_draw =0
		for i in range(0,120): # THIS IS THE SEQUENCE SIZE
			image_to_draw = images[id_image_draw+i,:]
			image_to_draw = (image_to_draw-np.min(image_to_draw))/(np.max(image_to_draw) - np.min(image_to_draw))
			image_to_draw = (image_to_draw*255).astype(np.uint8)
			write_image(image_to_draw,0, 10.0, (steer[id_image_draw+i,0])*15,(gas[id_image_draw+i,0])*3,self._config,1,input_control[id_image_draw+i],i)
			time.sleep(0.1)
	"""


	def update_log_state(self,i,duration):

		feedDict = self._training_manager.get_feed_dict()

		if hasattr(self._config, 'number_images_epoch'):
		    epoch_number = (float(i)*float(self._batch_size))/float(self._config.configInput().number_images_epoch)
		else:
			epoch_number = 0

		num_examples_per_step = self._batch_size 
		examples_per_sec = num_examples_per_step / duration
		train_accuracy = self._sess.run(self._training_manager.get_loss(), feed_dict=feedDict)
		reduced_accuracy = sum(train_accuracy)/len(train_accuracy)
		#print reduced_accuracy
		if  reduced_accuracy[0] < self._lowest_error:
			self._lowest_error = reduced_accuracy[0]
			self._lowest_iter = i

		self._last_train_error = reduced_accuracy[0]
		self._current_velocity = examples_per_sec
		self._epoch_number = epoch_number


	

	def print_screen_track(self,i,duration):

		if self._curses:

			#terminal_draw_tool.draw_training_percentage(int(float(self._epoch_number)/float(self._config.n_epochs)),time.time() - self._training_start_time)
			terminal_draw_tool.clear()
			terminal_draw_tool.draw_line(1,['Epoch: ',' Step: ',' Images P/Sec: ',' STATUS: '],[str(self._epoch_number),str(i),str(self._current_velocity),'TRAINING'],[1,1,1,2])
			terminal_draw_tool.draw_line(2,['best TRAIN: ',' on iter: ',' best VAL: ',' on iter: ',' Last Loss: '],[str(self._lowest_error),str(self._lowest_iter),str(self._lowest_val_error),str(self._lowest_val_iter),str(self._last_train_error)],[2,1,2,1,1])
			

			#print self._last_batch_gts[0].shape
			#print self._last_batch_error[0].shape		
			#print self._last_batch_targets[0].shape

			names_to_draw = []
			for i in self._config.inputs_to_print:
				names_to_draw.append(i)
		
			for i in self._config.targets_to_print:
				names_to_draw.append(i+'_o')
				names_to_draw.append(i+'_gt')
				names_to_draw.append(i+'_e')


			
			vector_to_draw = []

			for i in range(len(self._last_batch_inputs)):
				vector_to_draw.append(self._last_batch_inputs[i])


			# First Inputs
			# THEN OUtPUT GT and Error

			for i in range(len(self._last_batch_targets)):


				vector_to_draw.append(self._last_batch_targets[i])
				vector_to_draw.append(self._last_batch_gts[i])
				vector_to_draw.append(self._last_batch_error[i])


			
			terminal_draw_tool.draw_line(3,names_to_draw,[' ']*len(names_to_draw),[1]*len(names_to_draw))
			terminal_draw_tool.draw_vector_n_col(4,vector_to_draw)

		else:
			print "Epoch: %s, Step: %s, Images/Sec: %s" % (str(self._epoch_number), str(i),str(self._current_velocity))
			print "Best Train: %s, on iter: %s, Best Val: %s, on iter: %s, Last Loss: %s" % (str(self._lowest_error),str(self._lowest_iter),str(self._lowest_val_error),str(self._lowest_val_iter),str(self._last_train_error))
			

	def write_energy_contrib(self):


		feedDict = self._training_manager.get_feed_dict()
		


		variable_energy_tensor_vec = self._training_manager.get_variable_energy()
		self._last_energy_contrib_vec =[]
		for j in range(len(variable_energy_tensor_vec)):
			self._last_energy_contrib_vec.append(self._sess.run(variable_energy_tensor_vec[j], feed_dict=feedDict))


	def write_general_loss(self):

		

		feedDict = self._training_manager.get_feed_dict()
		train_accuracy = self._sess.run(self._training_manager.get_loss(), feed_dict=feedDict)



		with   open(self._config.train_path_write +'loss_function', 'a+') as outfile_energy:
			energy = sum(train_accuracy)/len(train_accuracy)
			outfile_energy.write("%f\n" % energy[0])
		
		outfile_energy.close()
		



	def write_variable_error(self):

		self._last_batch_error= []
		self._last_batch_targets = []
		self._last_batch_gts = []
		self._last_batch_inputs = []
		#self._last_batch_energy = []

		feedDict = self._training_manager.get_feed_dict()
		feedDict[self._training_manager._dout] = [1]*len(self._config.dropout)

		#output_data =feedDict[self._training_manager._targets_data[0]] # ZERO IS STEERINg

		for i in self._config.inputs_to_print:

			
			self._last_batch_inputs.append(feedDict[self._training_manager._targets_data[self._config.targets_names.index(i)]])
		


		for i in self._config.targets_to_print:
			#run this




			index_target =self._config.branch_config[self._config.selected_branch].index(i)
			targets = self._sess.run(self.split_target[self._config.targets_to_print.index(i)][index_target], feed_dict=feedDict)

			self._last_batch_targets.append(targets)
			self._last_batch_gts.append(feedDict[self._training_manager._targets_data[self._config.targets_names.index(i)]])


		

		variables_loss_vec = self._training_manager.get_variable_error()
		"""
		outfile = open(self._config.train_path_write +'variable_errors', 'a+')
		outfile_clean = open(self._config.train_path_write +'variable_errors_clean', 'a+')
	    """
	 	
		error_vec = self._sess.run(variables_loss_vec[self._config.selected_branch], feed_dict=feedDict)

		for var in self._config.targets_to_print:
			index_error =self._config.branch_config[self._config.selected_branch].index(var)


			self._last_batch_error.append(error_vec[index_error])

		"""
			#if j <self._config.number_steering_branches:
			#	error_vec_clean = np.multiply(np.reshape(input_control[:,j+1],(self._batch_size,1)),error_vec)*(float(self._batch_size)/float(np.sum(input_control[:,j+1])+1))
			outfile_gt = open(self._config.train_path_write + "B_" + str(j) + self._config.branch_config[j][0] + '_gt', 'a+') 
			for k in range(0,len(output_data)):
				outfile_gt.write("%f\n" % output_data[k][0])
			#rint len(output_data)
			#print error_vec.shape
			for k in range(0,len(output_data)):
				outfile.write("%f " % error_vec[k])
				#outfile_clean.write("%f " % error_vec_clean[k])
			outfile.write("\n")
			#outfile_clean.write("\n")
			outfile_gt.close()
		"""#


	def write_summary(self,i):



		""" Write all the  individual variables energy contributions """
		self.write_energy_contrib()
		
		""" Write all the individual outputs for each network branch """




		#network_outputs =self._training_manager.get_network_output()

		#network_estimation_vec = np.zeros((self._batch_size,len(self._config.branch_config))) # four is the number of branches
		#for j in range(len(network_outputs)):


		#	network_estimation_vec[:,j] = np.reshape(self._sess.run(network_outputs[j],feed_dict=feedDict),(self._batch_size))



		""" Write Batch average Energy """
		self.write_general_loss()



		""" Write variable errors """
		self.write_variable_error()

