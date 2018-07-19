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


	def __init__(self,file_prefix,image_size2,image_size1,current_file_number=0,record_image=True,number_of_images=3,image_cut=[0,240]):

		self._number_of_images = number_of_images
		self._record_image_hdf5 = True

		# SET WHERE YOU ARE CUTTING THE IMAGE

		self._image_cut = image_cut

		self._number_images_per_file = 200
		self._file_prefix = file_prefix
		self._image_size2 =image_size2
		self._image_size1 = image_size1
		self._record_image = record_image


		# TODO SET THE NUMBER OF REWARDS + ACTIONS
		self._number_rewards = 51
		self._num_cams = 1

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

	# TODO: Add some timestamp
	def record(self,images,speed,steer,steer_noise):


		self._data_queue.put([images,speed,steer,steer_noise])

	@threaded
	def run_disk_writer(self):

		while True:
			data = self._data_queue.get()
			if self._data_queue.qsize() % 100 == 0:
				print "QSIZE:",self._data_queue.qsize()
			self._write_to_disk(data)

	def _write_to_disk(self,data):
		
		images  = data[0]
		speed = data[1]
		steer = data[2]
		steer_noise = data[3]

		for i in range(self._num_cams):
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
			self.data_rewards[pos,0]  = steer

			self.data_rewards[pos,5]  = steer_noise
			self.data_rewards[pos,5]  = speed  
       
       

		
			#print "GAS - >", self.data_rewards[pos,1]


			#outfile = open(self._csv_writing_file,'a+')
			#outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,,%f,%f,%d,%d\n" %(capture_time,measurements.gps_lat,measurements.gps_long,measurements.gps_alt,measurements.mag_heading,measurements.vrf_hud_groundspeed,measurements.vrf_hud_throttle,measurements.gps_linear_vel_x,\
	 		#measurements.gps_linear_vel_y,measurements.gps_linear_vel_z,actions[i].steer,actions[i].gas,0.0,0.0,measurements.direction,i))
			#outfile.close()
			self._current_pos_on_file +=1

	def close(self):

		self._current_hf.close()



