"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import tensorflow as tf
from random import *
import sys

sys.path.append('../utils')
#from drawing_tools import *
#from check_image_outputs_cv import *
from skimage.transform import resize

#import terminal_draw_tool
from codification import *



class ValidationManager(object):


	def __init__(self,config,training_manager,sess,batch_tensor,merged_summary):


		self._training_manager = training_manager
		self._sess = sess # the session from tensorflow that is going to be used
		self._config = config
		self._batch_tensor = batch_tensor
		self._merged = merged_summary		
		self._val_writer = tf.summary.FileWriter(self._config.val_path_write,self._sess.graph)

	def load_dict(self,batch):

		feedDict = {self._training_manager._input_images:batch[0]}

		count=1
		
		for i in range(len(self._config.targets_names)):

			feedDict.update({self._training_manager._targets_data[i]:batch[count]})
			count+= 1



		for i in range(len(self._config.inputs_names)):

			feedDict.update({self._training_manager._input_data[i]:batch[count]})	
			count+= 1


		feedDict.update({self._training_manager._dout:[1]*len(self._config.dropout)})

		return feedDict




	def run(self,iter_number):



		capture_time = time.time()

		variables_loss_vec = self._training_manager.get_variable_error()

		sumError = 0
		sumEnergy = 0
		outfile = open(self._config.val_path_write +'variable_errors_val', 'a+')
		#outfile_clean = open(self._config.val_path_write +'variable_errors_val_clean', 'a+')
		
		number_of_batches = self._config.number_images_val/(self._config.batch_size_val)
		
		super_batch = [
				np.zeros((number_of_batches*self._config.batch_size_val,self._config.image_size[0],self._config.image_size[1],self._config.image_size[2]))


		]
		for i in range(len(self._config.targets_sizes)):
			super_batch += [np.zeros((number_of_batches*self._config.batch_size_val,self._config.targets_sizes[i]))]
			
		for i in range(len(self._config.inputs_sizes)):
			super_batch += [np.zeros((number_of_batches*self._config.batch_size_val,self._config.inputs_sizes[i]))]


		for j in range(0,number_of_batches):
		
			batch_val = self._sess.run(self._batch_tensor)
			feedDictVal = self.load_dict(batch_val)
			capture_time = time.time()
			# Load images on a supper batch in order to compute a higher number of validation images for tensorboard
			for k in range(len(super_batch)):
				#print super_batch[k][(j*self._config.batch_size_val):((j+1)*self._config.batch_size_val)].shape
				#print batch_val[k].shape
				super_batch[k][(j*self._config.batch_size_val):((j+1)*self._config.batch_size_val)] =batch_val[k]


			for k in range(0,len(self._config.branch_config)):


				outfile_gt = open(self._config.val_path_write +self._config.variable_names[k] +'_gt_val', 'a+')
				validation_result_error_vec= self._sess.run(variables_loss_vec[k], feed_dict=feedDictVal)

				#sumError =  sumError +validation_result_error_vec
				#if k <4:
				#validation_result_error_vec_clean = np.multiply(np.reshape \
				#	(batch_val[3][:,k+1],(self._batch_size,1)),validation_result_error_vec)*(120/float(np.sum(batch_val[3][:,k+1])+1))

				#for l in range(0,len(validation_result_error_vec)):
					#print 'wrote val for ', config.variable_names[k]
				#	outfile.write("%f " % validation_result_error_vec[l])
					#outfile_clean.write("%f " % validation_result_error_vec_clean[l])

				#	outfile_gt.write("%f\n" % batch_val[1][l][0])

				outfile_gt.close()
				#outfile.write("\n")
				#outfile_clean.write("\n")


			energy_val= self._sess.run(self._training_manager.get_loss(), feed_dict=feedDictVal)


			sumEnergy += sum(energy_val)


			with open(self._config.val_path_write +'loss_function_val', 'a+') as outfile_energy:
				for l in range(0,len(energy_val)):
					outfile_energy.write("%f\n" % energy_val[l])


			#outfile_energy.close()

		feedDict_cat = self.load_dict(super_batch)
		#if sumEnergy/self._config.number_images_epoch_val < self._lowest_val_error:
		""" This could  be used to se the log manager for later ploting """ 
		#	self._lowest_val_error = sumEnergy/self._config.number_images_epoch_val
		#	self._lowest_val_iter = i


		sumEnergy/self._config.number_images_val
		#difference = abs(current_val_error - self._average_val_error)


		outfile.close()
		#outfile_clean.close()

		summary = self._sess.run(self._merged,feed_dict = feedDict_cat)

		self._val_writer.add_summary(summary,iter_number)


		return time.time() - capture_time