import h5py
import scipy
import time
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


	def __init__(self,file_prefix,image_size2,image_size1,current_file_number=0,record_image=True,number_of_images=1,image_cut=[0,240]):

		self._number_of_images = number_of_images
		self._record_image_hdf5 = True
		self._image_cut = image_cut
		self._number_images_per_file = 200
		self._file_prefix = file_prefix
		self._image_size2 =image_size2
		self._image_size1 = image_size1
		self._record_image = record_image
		self._number_rewards = 51

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

	def record(self,images,rewards,action,action_noise):


		self._data_queue.put([images,rewards,action,action_noise])

	@threaded
	def run_disk_writer(self):

		while True:
			data = self._data_queue.get()
			if self._data_queue.qsize() % 100 == 0:
				print "QSIZE:",self._data_queue.qsize()
			self._write_to_disk(data)

	def _write_to_disk(self,data):
		
		images  = data[0]
		measurements = data[1]
		actions = data[2]
		action_noise = data[3]

		for i in range(self._number_of_images):
			if self._current_pos_on_file == self._number_images_per_file:
				self._current_file_number += 1
				self._current_pos_on_file = 0
				self._current_hf.close()
				self._current_hf = self._create_new_db()

			#print image.shape
			#print 'image cut',self._image_cut
			pos = self._current_pos_on_file
		
			capture_time   = int(round(time.time() * 1000))	
			
			#print int(round(time.time() * 1000))
			if self._record_image:
				im = Image.fromarray(images[i])
				im.save(self._images_writing_folder_vec[i] + str((capture_time)) + ".jpg")

			if self._record_image_hdf5:
				image = images[i][self._image_cut[0]:self._image_cut[1],:,:]
				#print images[i].shape
				#print self._image_cut
				image = scipy.misc.imresize(image,[self._image_size2,self._image_size1])
				self.data_center[pos] = image
			#print actions[i].steer 
			self.data_rewards[pos,0]  = actions[i].steer  
			self.data_rewards[pos,1]  = actions[i].gas 
			self.data_rewards[pos,2]  = actions[i].brake  
			self.data_rewards[pos,3]  = actions[i].hand_brake  
			self.data_rewards[pos,4]  = actions[i].reverse
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
			self.data_rewards[pos,44] = measurements.vrf_hud_airspeed
			self.data_rewards[pos,45] = measurements.vrf_hud_groundspeed
			self.data_rewards[pos,46] = measurements.vrf_hud_heading
			self.data_rewards[pos,47] = measurements.vrf_hud_throttle
			self.data_rewards[pos,48] = measurements.vrf_hud_altitude
			self.data_rewards[pos,49] = i
			self.data_rewards[pos,50] = capture_time
		
			#print "GAS - >", self.data_rewards[pos,1]


			outfile = open(self._csv_writing_file,'a+')
			outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,,%f,%f,%d,%d\n" %(capture_time,measurements.gps_lat,measurements.gps_long,measurements.gps_alt,measurements.mag_heading,measurements.vrf_hud_groundspeed,measurements.vrf_hud_throttle,measurements.gps_linear_vel_x,\
	 measurements.gps_linear_vel_y,measurements.gps_linear_vel_z,actions[i].steer,actions[i].gas,0.0,0.0,measurements.direction,i))
			outfile.close()
			self._current_pos_on_file +=1

	def close(self):

		self._current_hf.close()






