
import numpy as np
import math
import matplotlib.pyplot as plt


import argparse
sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
flatten = lambda l: [item for sublist in l for item in sublist]



def task_complete_percentages(data_matrix):

	complete_per = []
	pos_ant =0
	for pos in reset_positions:
		complete_per.append(sum(data_matrix[pos_ant:pos,1])/len(reset_positions))
		pos_ant =pos

	return complete_per

def task_average_time_percentages(data_matrix,reset_positions):

	complete_per = []
	pos_ant =0
	for pos in reset_positions:
		complete_per.append(sum(data_matrix[pos_ant:pos,0])/25.0)
		pos_ant =pos

	return complete_per


def get_colisions(selected_matrix):

	count_gen =0
	count_ped =0
	count_car = 0
	i=1

	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,4] - selected_matrix[(i-10),4]) > 40000:
			count_gen+=1
			i+=20
		i+=1


	i=1
	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,2] - selected_matrix[(i-10),2]) > 40000:
			count_car+=1
			i+=30
		i+=1


	i=1
	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,3] - selected_matrix[i-5,3]) > 30000:
			count_ped+=1
			i+=100
		i+=1


	return count_gen,count_car,count_ped

def get_distance_traveled(selected_matrix):

	prev_x = selected_matrix[0,0]
	prev_y = selected_matrix[0,1]

	i =1
	acummulated_distance =0
	while i < selected_matrix.shape[0]:

		x = selected_matrix[i,0]
		y = selected_matrix[i,1]
		

		acummulated_distance += sldist((x,y),(prev_x,prev_y))
		#print sldist((x,y),(prev_x,prev_y))

		prev_x =x
		prev_y =y

		i+=1
	return (float(acummulated_distance)/float(100*1000))

def get_end_positions(data_matrix):


	i=0
	end_positions_vec = []
	accumulated_time = 0
	while i < data_matrix.shape[0]:
		
		end_positions_vec.append(accumulated_time)
		accumulated_time += data_matrix[i,2]*10
		i+=1

	return end_positions_vec


def is_car_static(pos,reward_matrix):


	x = reward_matrix[pos,0]
	y = reward_matrix[pos,1]

	prev_x = reward_matrix[pos,0]
	prev_y = reward_matrix[pos,1]

	if sldist((x,y),(prev_x,prev_y)) > 100:
		return False
	else:
		return True






def get_end_positions_state(end_positions,data_matrix, reward_matrix):


	vector_of_infractions = [0,0,0,0] # Inf1+inf3 , inf2+inf3 or inf3,  , inf1+inf4, timeout


	for i in range(len(end_positions)):
		pos = int(end_positions[i] -20)

		if data_matrix[i,1] == 0: # if it failed, lets find the reason

			if reward_matrix[pos,4] > 30000 and is_car_static(pos,reward_matrix): # If it crashed_general

				if reward_matrix[pos,5] > 0.4: # Check if it is out of road
					# Case 0
					vector_of_infractions[0] +=1
				else: # Check it is out of lane or whaever
					vector_of_infractions[1] +=1






			elif reward_matrix[pos,2] > 30000 and is_car_static(pos,reward_matrix):


				if reward_matrix[pos,6] > 0.1: # Check if it is out of lane
					vector_of_infractions[2]+=1

				else:  # Likely that the it bumped the car but it didn't bother
					vector_of_infractions[3]+=1

			else:  # TimeOUt
				vector_of_infractions[3]+=1


	return vector_of_infractions


	


def get_out_of_road_lane(selected_matrix):

	count_road =0
	count_lane =0


	i=0

	while i < selected_matrix.shape[0]:
		#print selected_matrix[i,6]
		if (selected_matrix[i,5] - selected_matrix[(i-10),5]) > 0.3:
			count_road+=1
			i+=20
		
		if (selected_matrix[i,6] - selected_matrix[(i-10),6]) > 0.4:
			count_lane+=1
			i+=20

		i+=1


	

	return count_lane,count_road


def get_out_of_road_lane_per(selected_matrix):

	count_road =0
	count_lane =0


	i=0

	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,5] > 0):
			count_road+=1
		if (selected_matrix[i,6] > 0):
			count_lane+=1
		i+=1


	

	return float(count_lane)/float(selected_matrix.shape[0]),float(count_road)/float(selected_matrix.shape[0])



if __name__ == '__main__':


	path = ''
	parser = argparse.ArgumentParser(description='stats')
	# General
	parser.add_argument('files', metavar='files',default='', type=str, help='The file with results')


	args = parser.parse_args()
	files_parsed =  args.files.split(',')


	intervention_acc =[0,0,0,0,0]
	completions_acc = [0,0,0,0]
	infractions_vec = [0,0,0,0,0]
	for file in files_parsed:


		data_matrix = np.loadtxt(open(path + file, "rb"), delimiter=",", skiprows=1)


		reward_matrix = np.loadtxt(open(path + 'rewards_file_' + file, "rb"), delimiter=",", skiprows=1)


		completed_episodes = sum(data_matrix[:,2])/data_matrix.shape[0]
		print 'completions', completed_episodes

		#completions = task_complete_percentages(data_matrix)

		#completions_acc = [sum(x) for x in zip(completions, completions_acc)]


		km_run = get_distance_traveled(reward_matrix)

		#print 'colisions',get_colisions(reward_matrix[task_duration:,:])
		colisions = get_colisions(reward_matrix)


		print 'KM drove ',km_run
		lane_road = get_out_of_road_lane(reward_matrix)
		infractions = [lane_road[0]/km_run,lane_road[1]/km_run,colisions[0]/km_run,colisions[1]/km_run,colisions[2]/km_run]
		infractions_vec = [sum(x) for x in zip(infractions, infractions_vec)] 
		print 'Non_Colisions/Km', (infractions[1]+  infractions[0])/2.0 ,'Lane Cross/Km ',infractions[0],'Side Cross/Km ',infractions[1],'Col Gen /Km ',infractions[2]\
		,'Col Ped /Km ',infractions[3],'Col Ped /Km ',infractions[4], 'Acidents/Km ', (infractions[4] +infractions[2] + infractions[3])/3,\
		'total', 1/((infractions[4] +infractions[2] + infractions[3] + infractions[1] + infractions[0])/5.0)

			

