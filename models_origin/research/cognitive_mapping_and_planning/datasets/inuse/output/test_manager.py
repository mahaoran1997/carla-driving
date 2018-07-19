
import sys

sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla_interface')
sys.path.append('drive_interfaces/gta_interface')
sys.path.append('drive_interfaces/deeprc_interface')
sys.path.append('drive_interfaces/carla_interface/carla_client')

import tensorflow as tf
sys.path.append('drive_interfaces/carla_interface/carla_client/protoc')
import math 
import threading
sldist2 = lambda c1, c2: math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

import time


class TestManager(object):


	def __init__(self,config_test,config_input,sess,training_manager,port,exp_name):

		self._sess = sess
		self._config_input = config_input
		self._config_test = config_test
		self._exp_name  = exp_name

		driver_conf_module  = __import__(config_test.driver_config)
		driver_conf= driver_conf_module.configDrive()
		if port !=None:
			driver_conf.port = port

		self.episodes_positions = driver_conf.episodes_positions


		self.number_of_cars = driver_conf.number_of_cars
		self.number_of_pedestrians = driver_conf.number_of_pedestrians
		self.weathers  = driver_conf.weathers
		self.timeouts  = driver_conf.timeouts


		if config_test.interface_name == 'Carla':
			from carla_machine import CarlaMachine
			self._environment = CarlaMachine(trained_manager = training_manager\
				,driver_conf=driver_conf,session =sess,config_input=self._config_input)

		else:
			print " We only Suport carla"
		

		self._train_writer = tf.summary.FileWriter(self._config_input.test_path_write,self._sess.graph)

		self._completion = tf.placeholder("float",shape=())
		self._accidents = tf.placeholder("float",shape=())
		self._lane = tf.placeholder("float",shape=())
		self._sidewalk = tf.placeholder("float",shape=())
		self._ped = tf.placeholder("float",shape=())
		self._car = tf.placeholder("float",shape=())
		self._n_comp = tf.placeholder("float",shape=())
		self._speed = tf.placeholder("float",shape=())

		self._comp_sum = tf.summary.scalar('Completion', self._completion)
		self._acci_sum =tf.summary.scalar('Accidents P/Km', self._accidents)
		self._car_sum = tf.summary.scalar('Car Col/Km', self._car)
		self._ped_sum = tf.summary.scalar('Ped Col/Km', self._ped)
		self._sidewalk_sum=tf.summary.scalar('Time on Sidewalk', self._sidewalk)
		self._lane_sum =tf.summary.scalar('Time on Opp Lane', self._lane)
		self._n_comp_sum =tf.summary.scalar('Number of Total Completions', self._n_comp)
		self._speed_sum  =tf.summary.scalar('Average Speed', self._speed)

		#testing_thread = threading.Thread(target=self.simulation_loop, args=[self._sess])
		#testing_thread.isDaemon()
		#testing_thread.start()

	def simulation_loop(self,sess):
		iteration =0
		self._environment.start()
		while True:
			self.perform_simulation_test(iteration)

			iteration +=1

	def perform_simulation_test(self,iteration):

		#print " PERFORMING"
		
		self._environment.start()
		epi_rewards =[]
		epi_init_goals =[]
		total_iterations = 0.0
		total_speed = 0.0
		for i in range(len(self.episodes_positions)):
			# Start The long and required new episode.
			self._environment.new_episode(self.episodes_positions[i][0],self.episodes_positions[i][1],\
				self.number_of_cars[i],self.number_of_pedestrians[i],\
				self.weathers[i])


			# Get an initial Timestamp
			data = self._environment.carla.getReward()
			t0 = data[0].game_timestamp

			# initial init/end positions TODO: check the distance based on the path between two points ( PLANNER FEAT)
			init_end=[[data[0].player_x,data[0].player_y]]
			
			time_out = self.timeouts[i]
			finished = False 

			# Run steps on the environment until 
			rewards =[]
			while not finished:

				inst_reward = self._run_step(self.episodes_positions[i][1])




				goal_dist = self._get_distance_to_goal([inst_reward.player_x,inst_reward.player_y]\
					,self.episodes_positions[i][1])
				#print 'GOAL ',goal_dist
				if goal_dist < 200.0:
					finished=True


				t1 = inst_reward.game_timestamp
				#print inst_reward.game_timestamp
				#print t1 - t0
				if (t1-t0) > (time_out*1000):
					finished = True
				total_iterations+=1.0

				total_speed += inst_reward.speed
				rewards.append(inst_reward)


			init_end.append([inst_reward.player_x,inst_reward.player_y])
			init_end.append([self._environment.positions[self.episodes_positions[i][1]][0]\
			,self._environment.positions[self.episodes_positions[i][1]][1]])

			epi_rewards.append(rewards)
			epi_init_goals.append(init_end)



		# So far ( Percentage of completion, Number of accidents )
		#self._save_raw_rewards(epi_rewards)
		traveled_distance = float(self._get_traveled_dist(epi_rewards))
		col_gen,col_car,col_ped =self._get_colisions(epi_rewards)
		count_lane,count_side = self._get_out_of_road_lane(epi_rewards)
		percentage,number_completions =self._get_percentage_completion(epi_init_goals)

		self._write(iteration,percentage,(col_gen+col_car+col_ped)/traveled_distance,\
		 count_side/total_iterations,count_lane/total_iterations,col_ped/traveled_distance,\
		 col_car/traveled_distance,number_completions,total_speed/total_iterations)

		self._environment.stop()
		time.sleep(0.1)


	# From here we have just like ocluded functions

	def _run_step(self,target):

		rewards,images =self._environment.get_sensor_data()

		direction = self._environment.compute_direction((rewards.player_x,rewards.player_y,22),\
			(rewards.ori_x,rewards.ori_y,rewards.ori_z))

		action = self._environment.compute_action(images[0],rewards.speed,direction)
		self._environment.act(action[0])
		#print direction
		#print action[0].steer
		return rewards

	def _get_distance_to_goal(self,vec,goal_index):

		vec_goal = [self._environment.positions[goal_index][0]\
		,self._environment.positions[goal_index][1]]
		return sldist2(vec,vec_goal)

	def _get_out_of_road_lane(self,epi_rewards):

		count_side =0
		count_lane =0



		for rewards  in epi_rewards:

			i=0
			while i < len(rewards):
				if (rewards[i].road_intersect) > 0.2:
					count_lane+=1

				if (rewards[i].sidewalk_intersect) > 0.3:
					count_side+=1

				i+=1


		
		return count_lane,count_side		

	def _get_percentage_completion(self,epi_init_end_pos):
		# Check the rewardsfiles and get how much was it completed on each episode

		average_completion =0
		number_completions = 0
		for init_end_pos in epi_init_end_pos:

			distance_begin = sldist2(init_end_pos[0],init_end_pos[2])
			distance_end = sldist2(init_end_pos[1],init_end_pos[2])


			if (distance_end) < 210.0:
				number_completions+=1

			average_completion += (distance_begin - distance_end)/distance_begin


		average_completion/=len(epi_init_end_pos)
		#print average_completion
		return average_completion,number_completions



	def _get_traveled_dist(self,epi_rewards):



		acummulated_distance =0

		for rewards in epi_rewards:
			prev_x = rewards[0].player_x
			prev_y = rewards[0].player_y

			for i in range(1,len(rewards)):
				x = rewards[i].player_x
				y = rewards[i].player_y
		

				acummulated_distance += sldist2((x,y),(prev_x,prev_y))
				#print sldist((x,y),(prev_x,prev_y))

				prev_x =x
				prev_y =y

		return (float(acummulated_distance)/float(100*1000))

	def _get_colisions(self,epi_rewards):

		count_gen =0
		count_ped =0
		count_car = 0
		
		
		for rewards  in epi_rewards:
			i=1
			while i < len(rewards):
				if (rewards[i].collision_gen - rewards[i-10].collision_gen) > 40000:
					count_gen+=1
					i+=20
				i+=1


			i=1
			while i < len(rewards):
				if (rewards[i].collision_car - rewards[i-10].collision_car) > 40000:
					count_car+=1
					i+=30
				i+=1


			i=1
			while i < len(rewards):
				if (rewards[i].collision_ped - rewards[i-5].collision_ped) > 30000:
					count_ped+=1
					i+=100
				i+=1


		return [count_gen,count_car,count_ped]





	def _write(self,iteration,average_completion,accidents_per_km,sidewalk,lane,\
		ped_per_km,car_per_km,number_of_completions,average_speed):
		# Write on a tensorboard topic this result

		print "Writing:"
		print iteration
		print "Avg Com ",average_completion
		print "Acc per KM ",accidents_per_km
		print "Sidewalk time ",sidewalk
		print "lane ",lane
		print "Ped Per Km ",ped_per_km
		print "car per km ",car_per_km
		print " Number of comp ",number_of_completions
		print "Avg Speed ",average_speed

		outfile = open(self._exp_name + '.csv','a+')
		outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f\n" %(iteration,average_completion,accidents_per_km,sidewalk,lane,\
			ped_per_km,car_per_km,number_of_completions,average_speed))
		outfile.close()
		# The percentage of distance traveled by the model on all tasks
		summary1 = self._sess.run(self._comp_sum,feed_dict ={self._completion:average_completion} )
		# Number of general accidents per kilometer run
		summary2 = self._sess.run(self._acci_sum,feed_dict = {self._accidents:accidents_per_km})

		# percentage of time the model is on sidewalk
		summary3 = self._sess.run(self._sidewalk_sum,feed_dict = {self._sidewalk:sidewalk})
		# percentage of time the model is out of lane
		summary4 = self._sess.run(self._lane_sum,feed_dict = {self._lane:lane})
		
		# number of pedestrians hit per kilometer
		summary5 = self._sess.run(self._ped_sum,feed_dict = {self._ped:ped_per_km})


		# number of cars hit per kilometer
		summary6 = self._sess.run(self._car_sum,feed_dict = {self._car:car_per_km})

		# Number of times the models reachs to the end
		summary7 = self._sess.run(self._n_comp_sum,feed_dict = {self._n_comp:number_of_completions})

		# It is important to see how fast a model can go during training ( How does affect the speed the model goes)
		summary8 = self._sess.run(self._speed_sum,feed_dict = {self._speed:average_speed})


		self._train_writer.add_summary(summary1,iteration)
		self._train_writer.add_summary(summary2,iteration)
		self._train_writer.add_summary(summary3,iteration)
		self._train_writer.add_summary(summary4,iteration)
		self._train_writer.add_summary(summary5,iteration)
		self._train_writer.add_summary(summary6,iteration)
		self._train_writer.add_summary(summary7,iteration)
		self._train_writer.add_summary(summary8,iteration)	
