import h5py
import scipy
import time
from Pipe import Pipe
from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from PIL import Image
import numpy as np
import os

#lets put a big queue for the disk. So I keep it real time while the disk is writing stuff
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper

class Recorder(object):

	# We assume a three camera case not many cameras per input ....


	def __init__(self,file_prefix,image_size2,image_size1,current_file_number=0,record_image=True,number_of_images=3,image_cut=[0,180]):

		self._number_of_images = number_of_images
		self._record_image_hdf5 = True
		self._image_cut = image_cut
		self._number_images_per_file = 200
		self._file_prefix = file_prefix
		self._image_size2 =image_size2
		self._image_size1 = image_size1
		self._record_image = record_image
		self._number_rewards = 47

		if not os.path.exists(self._file_prefix):
			os.mkdir(self._file_prefix)


		self._current_file_number = current_file_number
		self._current_hf = self._create_new_db()


		self._images_writing_folder_vec =[]
		for i in range(number_of_images):
			images_writing_folder = self._file_prefix + "Images_" +str(i) + "/"
			if not os.path.exists(images_writing_folder):
				os.mkdir(images_writing_folder)
			self._images_writing_folder_vec.append(images_writing_folder)

		
		self._csv_writing_file = self._file_prefix + 'outputs.csv'
		self._current_pos_on_file =0 
		self._data_queue = Queue(8000)
		self.run_disk_writer()
		

	def _create_new_db(self):

		hf = h5py.File( self._file_prefix +'data_'+ str(self._current_file_number).zfill(5) +'.h5', 'w')
		self.data_center= hf.create_dataset('images_center', (self._number_images_per_file,self._image_size2,self._image_size1,3),dtype=np.uint8)
		#data_right= hf.create_dataset('images_right', (max_number_images_per_file,image_size2,image_size1,3),'f')
		self.data_rewards  = hf.create_dataset('targets', (self._number_images_per_file, self._number_rewards),'f')


		return hf

	def record(self,images,rewards,action,action_noise,folder_num):


		self._data_queue.put([images,rewards,action,action_noise,folder_num])

	@threaded
	def run_disk_writer(self):

		while True:
			data = self._data_queue.get()
			if self._data_queue.qsize() % 1000 == 0:
				print "QSIZE:",self._data_queue.qsize()
			self._write_to_disk(data)

	def _write_to_disk(self,data):
		if self._current_pos_on_file == self._number_images_per_file:
			self._current_file_number += 1
			self._current_pos_on_file = 0
			self._current_hf.close()
			self._current_hf = self._create_new_db()

		image  = data[0]
		measurements = data[1]
		action = data[2]
		action_noise = data[3]
		folder_num = data[4]
		#print image.shape
		#print 'image cut',self._image_cut
		pos = self._current_pos_on_file
		
		capture_time   = int(round(time.time() * 1000))	

		#print int(round(time.time() * 1000))
		if self._record_image:
			im = Image.fromarray(image)
			im.save(self._images_writing_folder_vec[folder_num] + str((capture_time)) + ".jpg")

		if self._record_image_hdf5:
			image = image[self._image_cut[0]:self._image_cut[1],:,:]
			image = scipy.misc.imresize(image,[self._image_size2,self._image_size1])
			self.data_center[pos] = image

		self.data_rewards[pos,0]  = action.steer  
		self.data_rewards[pos,1]  = action.gas 
		self.data_rewards[pos,2]  = action.brake  
		self.data_rewards[pos,3]  = action.hand_brake  
		self.data_rewards[pos,4]  = action.reverse
		self.data_rewards[pos,5]  = action_noise.steer  
		self.data_rewards[pos,6]  = action_noise.gas  
		self.data_rewards[pos,7]  = action_noise.brake 	        
		self.data_rewards[pos,8]  = measurements.direction 
		self.data_rewards[pos,9]  = measurements.gps_lat
		self.data_rewards[pos,10] = measurements.gps_long 
		self.data_rewards[pos,11] = measurements.gps_alt
		self.data_rewards[pos,12] = measurements.fused_linear_vel_x
		self.data_rewards[pos,13] = measurements.fused_linear_vel_z
		self.data_rewards[pos,14] = measurements.fused_linear_vel_y
		self.data_rewards[pos,15] = measurements.fused_angular_vel_x
		self.data_rewards[pos,16] = measurements.fused_angular_vel_y
		self.data_rewards[pos,17] = measurements.fused_angular_vel_z
		self.data_rewards[pos,18] = measurements.gps_linear_vel_x
		self.data_rewards[pos,19] = measurements.gps_linear_vel_y
		self.data_rewards[pos,20] = measurements.gps_linear_vel_z
		self.data_rewards[pos,21] = measurements.gps_angular_vel_x
		self.data_rewards[pos,22] = measurements.gps_angular_vel_y
		self.data_rewards[pos,23] = measurements.gps_angular_vel_z
		self.data_rewards[pos,24] = measurements.local_linear_vel_x
		self.data_rewards[pos,25] = measurements.local_linear_vel_y
		self.data_rewards[pos,26] = measurements.local_linear_vel_z
		self.data_rewards[pos,27] = measurements.local_angular_vel_x
		self.data_rewards[pos,28] = measurements.local_angular_vel_y
		self.data_rewards[pos,29] = measurements.local_angular_vel_z
		self.data_rewards[pos,30] = measurements.mag_heading
		self.data_rewards[pos,31] = measurements.imu_mag_field_x
		self.data_rewards[pos,32] = measurements.imu_mag_field_y
		self.data_rewards[pos,33] = measurements.imu_mag_field_z
		self.data_rewards[pos,34] = measurements.imu_angular_vel_x
		self.data_rewards[pos,35] = measurements.imu_angular_vel_y
		self.data_rewards[pos,36] = measurements.imu_angular_vel_z
		self.data_rewards[pos,37] = measurements.imu_linear_acc_x
		self.data_rewards[pos,38] = measurements.imu_linear_acc_y
		self.data_rewards[pos,39] = measurements.imu_linear_acc_z
		self.data_rewards[pos,40] = measurements.imu_orientation_a
		self.data_rewards[pos,41] = measurements.imu_orientation_b
		self.data_rewards[pos,42] = measurements.imu_orientation_c
		self.data_rewards[pos,43] = measurements.imu_orientation_d
		self.data_rewards[pos,44] = 0.0
		self.data_rewards[pos,45] = folder_num
		self.data_rewards[pos,46] = capture_time
		
		#print "GAS - >", self.data_rewards[pos,1]


		outfile = open(self._csv_writing_file,'a+')
		outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,,%f,%f,%d\n" %(capture_time,measurements.gps_lat,measurements.gps_long,measurements.gps_alt,measurements.mag_heading,0.0,0.0,measurements.fused_linear_vel_x,\
 measurements.fused_linear_vel_y,measurements.fused_linear_vel_z,action.steer,action.gas,0.0,0.0,measurements.direction))
		outfile.close()
		self._current_pos_on_file +=1

	def close(self):

		self._current_hf.close()






