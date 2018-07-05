
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


def get_colisions(selected_matrix,header):

	count_gen =0
	count_ped =0
	count_car = 0
	i=1

	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,header.index('collision_gen')] - selected_matrix[(i-10),header.index('collision_gen')]) > 40000:
			count_gen+=1
			i+=20
		i+=1


	i=1
	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,header.index('collision_car')] - selected_matrix[(i-10),header.index('collision_car')]) > 40000:
			count_car+=1
			i+=30
		i+=1


	i=1
	while i < selected_matrix.shape[0]:
		if (selected_matrix[i,header.index('collision_ped')] - selected_matrix[i-5,header.index('collision_ped')]) > 30000:
			count_ped+=1
			i+=100
		i+=1


	return count_gen,count_car,count_ped

def get_distance_traveled(selected_matrix,header):

	prev_x = selected_matrix[0,header.index('pos_x')]
	prev_y = selected_matrix[0,header.index('pos_y')]

	i =1
	acummulated_distance =0
	while i < selected_matrix.shape[0]:

		x = selected_matrix[i,header.index('pos_x')]
		y = selected_matrix[i,header.index('pos_y')]


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





def get_out_of_road_lane(selected_matrix,header):

	count_road =0
	count_lane =0


	i=0

	while i < selected_matrix.shape[0]:
		#print selected_matrix[i,6]
		if (selected_matrix[i,header.index('sidewalk_intersect')] - selected_matrix[(i-10),header.index('sidewalk_intersect')]) > 0.3:
			count_road+=1
			i+=20
		if i >= selected_matrix.shape[0]:
			break

		if (selected_matrix[i,header.index('lane_intersect')] - selected_matrix[(i-10),header.index('lane_intersect')]) > 0.4:
			count_lane+=1
			i+=20

		i+=1




	return count_lane,count_road


def print_infractions(infractions,km_run):
	ped_col = km_run
	car_col = km_run
	oth_col = km_run
	any_col = km_run
	sid_tre = km_run
	lan_tre = km_run
	any_tre = km_run
	any_inf = km_run

	if infractions[0] > 0:
		lan_tre = 1.0/infractions[0]

	if infractions[1] > 0:
		sid_tre = 1.0/infractions[1]

	if infractions[2] > 0:
		oth_col = 1.0/infractions[2]

	if infractions[3] > 0:
		car_col = 1.0/infractions[3]

	if infractions[4] > 0:
		ped_col = 1.0/infractions[4]

	if (infractions[0]+infractions[1]) > 0:
		any_tre = 1.0/(infractions[0]+infractions[1])
	
	if (infractions[2]+infractions[3]+infractions[4]) > 0:
		any_col = 1.0/(infractions[2]+infractions[3]+infractions[4])

	if (infractions[0]+infractions[1]+infractions[2]+infractions[3]+infractions[4]) > 0:
		any_inf = 1.0/(infractions[0]+infractions[1]+infractions[2]+infractions[3]+infractions[4])

	print '		Kilometers Without Going to Sidewalk - > ',sid_tre
	print '		Kilometers Without Crossing Lane - > ', lan_tre
	print '		Kilometers Without Pedestrians Collision - > ', ped_col
	print '		Kilometers Without Car Collision - > ', car_col
	print '		Kilometers Without Other Collision - > ', oth_col
	print '		----------------------------------------------------'
	print '		Kilometers Without Trespass ', any_tre
	print '		Kilometers Without Collision ', any_col
	print '		Kilometers Without Infraction ', any_inf
	print '		----------------------------------------------------'
	print '		Average Kilometers Per Trespass ', (sid_tre+lan_tre)/2
	print '		Average Kilometers Per Collision', (ped_col+car_col+oth_col)/3
	print '		Average Kilometers Per Infraction', (sid_tre+lan_tre+ped_col+car_col+oth_col)/5


if __name__ == '__main__':


	path = ''
	parser = argparse.ArgumentParser(description='stats')
	# General
	parser.add_argument('files', metavar='files',default='', type=str, help='The file with results')
  	parser.add_argument('-w', '--weather', default="1", type=int, help='The weathers')


	args = parser.parse_args()
	files_parsed =  args.files.split(',')

	intervention_acc =[0,0,0,0,0]
	completions_acc = [0,0,0,0]
	infractions_vec = [0,0,0,0,0]
	km_run_all = 0
	compute_infractions = True

	#summary_weathers = {'train_weather': [1,3,6,8]} #'test_weather': [4,14]
	#summary_weathers = {'train_weather': [2]} 
	summary_weathers = {'weather': [args.weather]} 

	#tasks =[0]
	#tasks =[0,1,2,3]
	#weathers = [1,3,6,8,2,14]
	for file in files_parsed:
		f = open(path + file, "rb")
		header = f.readline()
		header= header.split(',')
		header[-1] = header[-1][:-2]
		f.close()
		print header
		f = open(path + file + '_rewards_file', "rb")
		header_rewards = f.readline()
		header_rewards= header_rewards.split(',')
		header_rewards[-1] = header_rewards[-1][:-2]
		f.close()
		print header_rewards
		data_matrix = np.loadtxt(open(path + file, "rb"), delimiter=",", skiprows=1)

		tasks = np.unique(data_matrix[:,header.index('exp_id')])

		reward_matrix = np.loadtxt(open(path + file + '_rewards_file', "rb"), delimiter=",", skiprows=1)

		for t in tasks:
			task_data_matrix = data_matrix[data_matrix[:,header.index('exp_id')]== t]
			weathers = np.unique(task_data_matrix[:,header.index('weather')])
			summaries = {}
			for sw in summary_weathers:
				summaries[sw] = {'completion': 0., 'infractions': np.zeros(5, dtype=np.float), 'num_weathers': 0}
			for w in weathers:

				task_data_matrix  =data_matrix[np.logical_and(data_matrix[:,header.index('exp_id')]== t, data_matrix[:,header.index('weather')]== w)]
				if compute_infractions:
					task_reward_matrix =reward_matrix[np.logical_and(reward_matrix[:,header_rewards.index('exp_id')]== float(t), reward_matrix[:,header_rewards.index('weather')]== float(w))]

				completed_episodes = sum(task_data_matrix[:,header.index('result')])/task_data_matrix.shape[0]
				print 'Task ',t , 'Weather', w

				print '		Entire Episodes Completed (%) - > ', completed_episodes
				print ''

				#completions_acc = [sum(x) for x in zip(completions, completions_acc)]

				for sw in summary_weathers:
					if w in summary_weathers[sw]:
						summaries[sw]['completion'] += completed_episodes
						summaries[sw]['num_weathers'] += 1

				if compute_infractions:
					print '		==== Infraction Related ====='
					km_run = get_distance_traveled(task_reward_matrix,header_rewards)
					km_run_all = km_run_all + km_run
					print '		Drove (KM) - > ', km_run
					lane_road = get_out_of_road_lane(task_reward_matrix,header_rewards)
					colisions = get_colisions(task_reward_matrix,header_rewards)
					infractions = [lane_road[0]/km_run,lane_road[1]/km_run,colisions[0]/km_run,colisions[1]/km_run,colisions[2]/km_run]
					print_infractions(infractions,km_run)

					for sw in summary_weathers:
						if w in summary_weathers[sw]:
							# print summaries[sw]
							# print infractions
							summaries[sw]['infractions'] += np.array(infractions)

			print '\n\n >>> Task', t, 'summary <<<\n\n'
			for sw in summary_weathers:
				print sw, summary_weathers[sw]
				print 'Num weathers', summaries[sw]['num_weathers']
				print 'Drove (KM) - > ', km_run_all
				print 'Avg completion', summaries[sw]['completion']/summaries[sw]['num_weathers']
				print 'Avg infractions'
				print_infractions(summaries[sw]['infractions']/summaries[sw]['num_weathers'],km_run_all)
				#
				#
				#infractions_vec = [sum(x) for x in zip(infractions, infractions_vec)]
				#print 'Non_Colisions/Km', (infractions[1]+  infractions[0])/2.0 ,'Lane Cross/Km ',infractions[0],'Side Cross/Km ',infractions[1],'Col Gen /Km ',infractions[2]\
				#,'Col Ped /Km ',infractions[3],'Col Ped /Km ',infractions[4], 'Acidents/Km ', (infractions[4] +infractions[2] + infractions[3])/3,\
				#'total', 1/((infractions[4] +infractions[2] + infractions[3] + infractions[1] + infractions[0])/5.0)
