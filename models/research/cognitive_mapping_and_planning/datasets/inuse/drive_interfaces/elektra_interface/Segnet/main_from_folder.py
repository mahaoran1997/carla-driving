import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time
import socket
import string

image_path="/home/nvidia/stixels-master4/track_evening/run3/left/"
sys.path.append('/usr/local/lib/python2.7/site-packages')

# Make sure that caffe is on the python path:
caffe_root = '/home/SegNet-Tutorial-master/caffe-segnet-segnet-cleaned/'
sys.path.insert(0,caffe_root+ 'python')
print caffe_root
import caffe


path_output = '/home/nvidia/SegNet-Tutorial-master/Output-Images_track/'
total_rows = 360    #indexing from 0
total_cols = 480 
new_car_speed = 0
old_car_speed = 0  
global horizon = total_rows/3

text_file = open("Decision.txt", "w")


UDP_IP = "10.42.0.144"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP

def MoveCar(image, old_car_speed):
	flag=0
	for i in range(total_rows):
		for j in range(total_cols):
			if np.all(image[i,j] == [128, 64, 128]):
				horizon = i
				flag=1				
				break
		if flag==1:
			break

	level_1 = total_rows - ((total_rows - horizon)/3)
	level_2 = total_rows - ((total_rows - horizon)*2/3)
	#road_count_1 = 0
	#road_count_2 = 0
	#road_count_3 = 0
	obstacle_count_1 = 0
	obstacle_count_2 = 0
	obstacle_count_3 = 0
	flag1 = 0
	flag2 = 0
	flag3 = 0

	#Code implementing obstacle<0.05 for the three levels
	for i in range(level_1, total_rows):								#Level 1			
		for j in range(total_cols/2 -total_cols/10, total_cols/2 +total_cols/10):		
			if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):     #if no road or markings	
				obstacle_count_1 = obstacle_count_1 + 1
				if(float(obstacle_count_1)/float(0.2*total_cols*(total_rows-level_1))) > 0.05: #if obstacles more than 											5%, no need to compute further, car speed remains zero
					flag1 = 1
					break
		if flag1 == 1: 
			break	
	if float(obstacle_count_1)/float(0.2*total_cols*(total_rows-level_1)) < 0.05: 			#Level 1 clear, go ahead
		new_car_speed=10
		for i in range(level_2,level_1):							#Level 2			
			for j in range(total_cols/2 -(int)(0.4*total_cols)/5, total_cols/2 +(int)(0.4*total_cols)/5):		
				if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):     
					obstacle_count_2 = obstacle_count_2 + 1
					if(float(obstacle_count_2)/float(0.2*total_cols*(total_rows-level_1))) > 0.05: 
						flag2 = 1
						break
			if flag2 == 1: 
				break	
		if float(obstacle_count_2)/float(0.8*total_cols/5*(level_1-level_2)) < 0.05: 		#Level 2 clear, go ahead
			new_car_speed=20
			for i in range(horizon,level_2):						#Level 3			
				for j in range(total_cols/2 -(int)(0.2*total_cols)/5, total_cols/2 +(int)(0.2*total_cols)/5):		
					if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):     
						obstacle_count_3 = obstacle_count_3 + 1
						if(float(obstacle_count_3)/float(0.4*total_cols/5*(level_2-horizon))) > 0.05: 
							flag3 = 1
							break
				if flag3 == 1: 
					break	
			if float(obstacle_count_3)/float(0.4*total_cols/5*(level_2-horizon)) < 0.05: 	 #Level 3 clear, go ahead
				new_car_speed=30
	else:
		new_car_speed = 0


	''' #Code implementing road>0.95
	for i in range(level_1,total_rows):
		for j in range(total_cols/2 -total_cols/10, total_cols/2 +total_cols/10):
			if (np.all(image[i,j] == [128, 64, 128]) or np.all(image[i,j] == [0,69,255])):
				road_count_1 = road_count_1 + 1
	if float(road_count_1)/float((total_cols/5)*(total_rows-level_1)) > 0.95: 
		new_car_speed=10
		for i in range(level_2,level_1):
			for j in range(total_cols/2 -(int)(0.4*total_cols)/5, total_cols/2 +(int)(0.4*total_cols)/5):
				if (np.all(image[i,j] == [128, 64, 128]) or np.all(image[i,j] == [0,69,255])):
					road_count_2 = road_count_2 + 1
		if float(road_count_2)/float(0.8*total_cols/5*(level_1-level_2)) > 0.95: 
			new_car_speed=20
			for i in range(horizon,level_2):
				for j in range(total_cols/2 -(int)(0.2*total_cols)/5, total_cols/2 +(int)(0.2*total_cols)/5):
					if (np.all(image[i,j] == [128, 64, 128]) or np.all(image[i,j] == [0,69,255])):
						road_count_3 = road_count_3 + 1
			if float(road_count_3)/float(0.4*total_cols/5*(level_2-horizon)) > 0.95: 
				new_car_speed=30
	else: new_car_speed=0'''

	change=(new_car_speed-old_car_speed)/10
	
	if(change<0):
		print "Control for decrease"
		for k in range(abs(change)):		
			MESSAGE = '<'
			print MESSAGE
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))	
		MESSAGE = 'w'
		print MESSAGE
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))				
	
	
	else: 
		print "Control for increase"		
		for k in range(abs(change)):		
			MESSAGE = '>'
			print MESSAGE
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))	
		MESSAGE = 'w'
		print MESSAGE
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
	return new_car_speed
	



# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
args = parser.parse_args()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

caffe.set_mode_gpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

for filename in os.listdir(image_path):
	start = time.time()
	frame = cv2.imread(str(image_path+filename))
	end = time.time()
	print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'
	print str(image_path+filename)
	
	start = time.time()
	frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
	input_image = frame.transpose((2,0,1))
	# input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
	input_image = np.asarray([input_image])
	end = time.time()
	print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

	start = time.time()
	out = net.forward_all(data=input_image)
	end = time.time()
	print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

	start = time.time()
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

	cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
	segmentation_rgb = segmentation_rgb.astype(float)/255
	end = time.time()
	print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

	new_car_speed =MoveCar(segmentation_rgb*255, old_car_speed)	
	old_car_speed =new_car_speed

	cv2.imwrite(str(path_output+filename),segmentation_rgb*255)
	
	text_file.write("Image:%s   Speed:%s\n" %(filename, new_car_speed))

	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
	    break
	
text_file.close()
cv2.destroyAllWindows()


