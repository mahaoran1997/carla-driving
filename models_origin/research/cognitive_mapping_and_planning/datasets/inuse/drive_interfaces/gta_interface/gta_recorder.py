
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



	def __init__(self,file_prefix,image_size2,image_size1,current_file_number=0,record_image=True):

		self._number_images_per_file = 200
		self._file_prefix = file_prefix
		self._image_size2 =image_size2
		self._image_size1 = image_size1
		self._record_image = record_image
		self._number_rewards = 23

		if not os.path.exists(self._file_prefix):
			os.mkdir(self._file_prefix)


		self._current_file_number = current_file_number
		self._current_hf = self._create_new_db()


		self._images_writing_folder = self._file_prefix + "Images/"
		if not os.path.exists(self._images_writing_folder):
			os.mkdir(self._images_writing_folder)

		
		self._csv_writing_file = self._file_prefix + 'outputs.csv'
		self._current_pos_on_file =0 
		self._data_queue = Queue(2000)
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
			#print "QSIZE:",self._data_queue.qsize()
			self._write_to_disk(data)




	def _write_to_disk(self,data):
		if self._current_pos_on_file == self._number_images_per_file:
			self._current_file_number += 1
			self._current_pos_on_file = 0
			self._current_hf.close()
			self._current_hf = self._create_new_db()





		image  = data[0]
		rewards = data[1]
		action = data[2]
		action_noise = data[3]

		pos = self._current_pos_on_file

		
		capture_time   = int(round(time.time() * 1000))

		image = image[:, :, ::-1]

		#print int(round(time.time() * 1000))
		if self._record_image:
			im = Image.fromarray(image)
			im.save(self._images_writing_folder + str((capture_time)) + ".jpg")

		image = image[130:530,:,:]
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
		self.data_rewards[pos,8]  = rewards.position[0]
		self.data_rewards[pos,9]  = rewards.position[1]
		self.data_rewards[pos,10]  = rewards.position[2]

		self.data_rewards[pos,11]  = rewards.speed

		self.data_rewards[pos,12]  = rewards.yaw
		self.data_rewards[pos,13]  = rewards.yaw_rate
		self.data_rewards[pos,14]  = rewards.collided
		self.data_rewards[pos,15]  = rewards.lane
		self.data_rewards[pos,16]  = rewards.reseted
		self.data_rewards[pos,17]  = rewards.goal[0] 
		self.data_rewards[pos,18]  = rewards.goal[1] 
		self.data_rewards[pos,19]  = rewards.direction
		self.data_rewards[pos,20]  = rewards.dist_to_goal
		self.data_rewards[pos,21]  = rewards.noise
		self.data_rewards[pos,22]  = rewards.time_stamp
		#print "GAS - >", self.data_rewards[pos,1]



		#outfile =open(self._csv_writing_file,'a+')
		#outfile.write("%d,%f,%f,%d,%f,%f,%f,%f,%f,%f,%f,%d,%d\n" % (capture_time ,action[0]\
		#	,action[1],int(direction),speed,yaw_rate,car_pos[0],car_pos[1],car_goal[0],car_goal[1],lane,collided,reset))			
		#outfile.close()


		self._current_pos_on_file +=1

	def close(self):

		self._current_hf.close()






