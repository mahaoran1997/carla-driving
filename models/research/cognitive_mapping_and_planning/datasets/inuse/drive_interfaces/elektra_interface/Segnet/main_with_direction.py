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


sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/SegNet-Tutorial-master/caffe-segnet-segnet-cleaned/'
sys.path.insert(0,caffe_root+ 'python')
print caffe_root
import caffe

path_input = '/home/nvidia/SegNet-Tutorial-master/Input-Images/'
path_output = '/home/nvidia/SegNet-Tutorial-master/Output-Images/'
i = 1
total_rows = 360 
total_cols = 480 
new_car_speed = 0
old_car_speed = 0  

text_file = open("Decision_with_direction.txt", "w")

UDP_IP = "10.42.0.144"
UDP_PORT = 5050
sock = socket.socket(socket.AF_INET, # Internet
					socket.SOCK_DGRAM) # UDP

def MoveCar(image, old_car_speed):

	##### Code for speed 
	flag=0
	horizon = -1
	for i in range(total_rows):
		for j in range(total_cols):
			if np.all(image[i,j] == [128, 64, 128]):
				horizon = i
				flag=1				
				break
		if flag==1:
			break
	if i == total_rows-1 and j == total_cols-1 and horizon == -1 :
		horizon = total_rows/3

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

	#Calculating forward speed, Code implementing obstacle<0.05 for the three levels
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
	

	#### Code for direction
	section_1_count = 0  									#keeps count of obstacle
	section_2_count = 0 
	section_4_count = 0
	section_5_count = 0 
	new_rotation = 0									#positive value implies right turn
	for i in range(level_1, total_rows):							#Section 2
		for j in range(int(round(0.2*total_cols)), int(round(0.4*total_cols))):		#1/5th of image width			
			if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):     #if no road or markings	
				section_2_count = section_2_count + 1
	if float(section_2_count)/float(0.2*total_cols*(total_rows-level_1)) > 0.75: 
		new_rotation = 60
	else:
		for i in range(level_1, total_rows):						#Section 1
			for j in range(int(round(0.2*total_cols))):							
				if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):     
					section_1_count = section_1_count + 1
		if float(section_1_count)/float(0.2*total_cols*(total_rows-level_1)) > 0.75: 
			new_rotation = 30

	for i in range(level_1, total_rows):							#Section 4
		for j in range(int(round(0.6*total_cols)), int(round(0.8*total_cols))):						
			if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):    
				section_4_count = section_4_count + 1
	if float(section_4_count)/float(0.2*total_cols*(total_rows-level_1)) > 0.75: 
		new_rotation = new_rotation - 60
	else:
		for i in range(level_1, total_rows):						#Section 5
			for j in range(int(round(0.8*total_cols)), total_cols):						
				if(not(np.all(image[i,j] == [128,64,128]) or np.all(image[i,j] == [0,69,255]))):
					section_5_count = section_5_count + 1
		if float(section_5_count)/float(0.2*total_cols*(total_rows-level_1)) > 0.75: 
			new_rotation = new_rotation - 30
	
	print "Direction: %s" %new_rotation


	#Setting car speed during rotation, sending direction to pi
	if new_rotation == 0:
		MESSAGE = 'x';									
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
	elif new_rotation == 30:
		MESSAGE = 'd';									
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		new_car_speed = 10
	elif new_rotation == 60:
		MESSAGE = 'd';									
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		new_car_speed = 20					#obstacle very near, turn fast
	elif new_rotation == -30:
		MESSAGE = 'a';									
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		new_car_speed = 10
	elif new_rotation == -60:
		MESSAGE = 'a';									
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		new_car_speed = 20


	#Sending car speed to pi
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


	'''#Setting steering angle
	if(new_rotation * old_rotation) <= 0:							#Change in direction
		MESSAGE = 'x';									#Reset steering angle to 0
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		for k in range(abs(new_rotation)/10):						#Set magnitude(absolute) 
			MESSAGE = 'l';
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		if new_rotation > 0:								#Set direction
			MESSAGE = 'd';
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		else:
			MESSAGE = 'a';
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
	else:											#No change in direction, update the angle
		if (abs(new_rotation) - abs(old_rotation)) > 0:
			#print "Control for decrease"
			for k in range((abs(new_rotation) - abs(old_rotation))/10):		
				MESSAGE = 'l';
				sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		else:
			for k in range((abs(old_rotation) - abs(new_rotation))/10):		
				MESSAGE = 'k';
				sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))'''	



	''' #Code for relative angles
	if (abs(new_rotation) - abs(old_rotation)) > 0:
		#print "Control for decrease"
		for k in range((abs(new_rotation) - abs(old_rotation))/10):		
			MESSAGE = 'l';
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
	else:
		for k in range((abs(old_rotation) - abs(new_rotation))/10):		
			MESSAGE = 'k';
			sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

	if  new_rotation> 0:
		#print "Control for decrease"	
		MESSAGE = 'd';
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
	else:
		#print "Control for decrease"	
		MESSAGE = 'a';
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT)	'''

	return new_car_speed, new_rotation


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

cv2.namedWindow("Input")
cv2.namedWindow("SegNet")

#cap = cv2.VideoCapture(0) # Change this to your webcam ID, or file name for your video file
cap = cv2.VideoCapture(1)
rval = True

while rval:
	start = time.time()
	rval, frame = cap.read()
	r=frame.shape[0]
	c=frame.shape[1]
	frame = frame[1:r/2, 1:c/2]
	
        if rval == False:
            break
	
	end = time.time()
	print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

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
	print 'Shape of segmentation_rgb', segmentation_rgb.shape
	end = time.time()
	print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

	new_car_speed, new_rotation = MoveCar(segmentation_rgb*255, old_car_speed)	
	old_car_speed = new_car_speed
	cv2.imshow("Input", frame)
	cv2.imshow("SegNet", segmentation_rgb)
	
	cv2.imwrite(str(path_input) + str(i) + ".png",frame)
	cv2.imwrite(str(path_output) + str(i) + ".png",segmentation_rgb*255)
	i = i+1
	
	text_file.write("Image:%s   Speed:%s   Direction:%s\n" %(i, new_car_speed,new_rotation))

	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
	    break
text_file.close()
cap.release()
cv2.destroyAllWindows()


