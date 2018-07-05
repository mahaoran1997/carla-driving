import math
import numpy as np
from carla.planner import Waypointer
from carla.protoc import Control

class Agent(object):

	# The
	def __init__(self,config):

		self.waypointer = Waypointer(config)
		#Default Throttle
		self.throttle = config.throttle #0.5
		#Default Brake
		self.brake = config.brake #0
		#Gain on computed steering angle
		self.steer_gain = config.steer_gain #0.8
		#Distance Threshold Traffic Light
		self.tl_dist_thres = config.tl_dist_thres #1800
		#Angle Threshold Traffic Light
		self.tl_angle_thres = config.tl_angle_thres #0.6
		#Distance Threshold Pedestrian
		self.p_dist_thres = config.p_dist_thres #1200
		#Angle Threshold Pedestrian
		self.p_angle_thres = config.p_angle_thres #0.3
		#Distance Threshold Vehicle
		self.v_dist_thres = config.v_dist_thres #1500
		#Angle Threshold Vehicle
		self.v_angle_thres = config.v_angle_thres #0.2
		#Select WP Number
		self.wp_num_steer = config.wp_num_steer #Select WP - Reverse Order: 1 - closest, 0 - furthest
		self.wp_num_speed = config.wp_num_speed #Select WP - Reverse Order: 1 - closest, 0 - furthest
		#Stop for traffic lights
		self.stop4TL = config.stop4TL #True
		#Stop for pedestrians
		self.stop4P = config.stop4P #True
		#Stop for vehicles
		self.stop4V = config.stop4V #True
		#Strength for applying brake - Value between 0 and 1
		self.brake_strength = config.brake_strength #1 
		#Factor to control coasting
		self.coast_factor = config.coast_factor #2
		#PID speed controller
		self.pid = config.pid
		#Target speed - could be controlled by speed limit
		self.target_speed = config.target_speed
		#Throttle max
		self.throttle_max = config.throttle_max
		#Flag to decide if PID should be used
		self.usePID = config.usePID

		self.speed_factor = 1		
		self.current_speed = 0
		self.wp = [0,0,0]
		self.wp_speed = [0,0,0] 


	def controller(self,wp_angle,wp_mag,wp_angle_speed):
		control = Control()

		steer = self.steer_gain*wp_angle
		if steer > 0:
		  control.steer = min(steer,1)
		else:
		  control.steer = max(steer,-1)
		
		if (self.usePID):
			# Don't go to fast around corners
			if math.fabs(wp_angle_speed) < 0.1:
			  target_speed_adjusted = self.target_speed * self.speed_factor
			elif math.fabs(wp_angle_speed) < 0.5:
			  target_speed_adjusted = 25 * self.speed_factor
			else:
			  target_speed_adjusted = 20 * self.speed_factor
		
			self.pid.target = target_speed_adjusted
			pid_gain = self.pid(feedback=self.current_speed)
			print ('Target: ', self.pid.target,'Error: ',self.pid.error,'Gain: ',pid_gain)
			print ('Target Speed: ', target_speed_adjusted, 'Current Speed: ', self.current_speed, 'Speed Factor: ', self.speed_factor)

			self.throttle = min (max (self.throttle - 0.25*pid_gain,0),self.throttle_max) 

			if pid_gain > 0.5:
			  self.brake = min (0.25*pid_gain*self.brake_strength,1)
			else:
			  self.brake = 0
		else:
			throttle_adjusted = self.throttle_max
			brake_adjusted = 0

			# Don't go to fast around corners
			if (math.fabs(wp_angle_speed) < 0.1):
			  throttle_adjusted = self.speed_factor * self.throttle_max
			elif (math.fabs(wp_angle_speed) < 0.5) and self.current_speed > 20:
			  throttle_adjusted = 0.8 * self.speed_factor * self.throttle_max
			  brake_adjusted = (1-0.8)*(1-self.speed_factor)*self.brake_strength
			elif self.current_speed > 15:
			  throttle_adjusted = 0.6 * self.speed_factor * self.throttle_max
			  brake_adjusted = (1-0.6)*(1-self.speed_factor)*self.brake_strength

			if (self.current_speed > self.target_speed):
			  throttle_adjusted = (1/(self.current_speed+2 - self.target_speed) * self.throttle_max + 2*self.throttle_max/3) * self.speed_factor
			if (self.speed_factor < 1 and self.current_speed > 5):
			  brake_adjusted = 0.5*(1-self.speed_factor)*self.brake_strength


			self.throttle = min (max (throttle_adjusted,0),self.throttle_max) 
			self.brake = min (max (brake_adjusted,0),1)

		
		control.throttle = max(self.throttle,0.01) #Prevent N by putting at least 0.01
		control.brake = self.brake


		print ('Throttle: ', control.throttle, 'Brake: ', control.brake, 'Steering Angle: ', control.steer)

		return control
		# TODO FUNCTION receives waypoints or whateverneeds and returns a control


	# receives the measurements and returns a control 


	def get_vec_dist (self, x_dst, y_dst, x_src, y_src):
	  vec = np.array([x_dst,y_dst] - np.array([x_src, y_src]))
	  dist = math.sqrt(vec[0]**2 + vec[1]**2)
	  return vec/dist, dist
	
	def get_angle (self, vec_dst, vec_src):
	  angle = math.atan2(vec_dst[1],vec_dst[0]) - math.atan2(vec_src[1],vec_src[0])
	  if angle > math.pi:
	    angle -= 2*math.pi
	  elif angle < -math.pi:
	    angle += 2*math.pi
	  return angle


	def get_control(self,measurements,target):
		# Reset speed factor		
		self.speed_factor = 1
		speed_factor_tl = 1
		speed_factor_tl_temp = 1
		speed_factor_p = 1
		speed_factor_p_temp = 1
		speed_factor_v = 1
		speed_factor_v_temp = 1
		player = measurements['PlayerMeasurements']
		agents = measurements['Agents']
		#print ' it has  ',len(agents),' agents'
		loc_x_player = player.transform.location.x
		loc_y_player = player.transform.location.y
		ori_x_player = player.transform.orientation.x
		ori_y_player = player.transform.orientation.y


		waypoints = self.waypointer.get_next_waypoints((player.transform.location.x,player.transform.location.y,22)\
            ,(player.transform.orientation.x,player.transform.orientation.y,player.transform.orientation.z)\
            ,(target.location.x,target.location.y,target.location.z)\
            ,(1,0,0))
		if waypoints == []:
			waypoints = [[player.transform.location.x,player.transform.location.y,22]]

		self.wp = [waypoints[int(self.wp_num_steer*len(waypoints))][0], waypoints[int(self.wp_num_steer*len(waypoints))][1]]
                wp_vector, wp_mag = self.get_vec_dist(self.wp[0],self.wp[1], loc_x_player, loc_y_player)

		if wp_mag>0:
			wp_angle = self.get_angle (wp_vector, [ori_x_player,ori_y_player])
		else:
			wp_angle = 0

		#WP Look Ahead for steering
		self.wp_speed = [waypoints[int(self.wp_num_speed*len(waypoints))][0], waypoints[int(self.wp_num_speed*len(waypoints))][1]]
                wp_vector_speed, wp_mag_speed = self.get_vec_dist(self.wp_speed[0],self.wp_speed[1], loc_x_player, loc_y_player)
		wp_angle_speed = self.get_angle (wp_vector_speed, [ori_x_player,ori_y_player])

		#print ('Next Waypoint (Steer): ', waypoints[self.wp_num_steer][0], waypoints[self.wp_num_steer][1])
		#print ('Car Position: ', player.transform.location.x, player.transform.location.y)
		#print ('Waypoint Vector: ', wp_vector[0]/wp_mag, wp_vector[1]/wp_mag)
		#print ('Car Vector: ', player.transform.orientation.x, player.transform.orientation.y)
		#print ('Waypoint Angle: ', wp_angle, ' Magnitude: ', wp_mag)

		for agent in agents:
		    if agent.HasField('traffic_light') and self.stop4TL:
		        if agent.traffic_light.state !=0: #Not green
		          x_agent = agent.traffic_light.transform.location.x
		          y_agent = agent.traffic_light.transform.location.y
		          tl_vector, tl_dist = self.get_vec_dist(x_agent, y_agent, loc_x_player, loc_y_player)
		          #tl_angle = self.get_angle(tl_vector,[ori_x_player,ori_y_player])
			  tl_angle = self.get_angle(tl_vector,wp_vector)
		          #print ('Traffic Light: ', tl_vector, tl_dist, tl_angle)
			  if (0 < tl_angle < self.tl_angle_thres/self.coast_factor and tl_dist < self.tl_dist_thres*self.coast_factor) or (0 < tl_angle < self.tl_angle_thres and tl_dist < self.tl_dist_thres) and math.fabs(wp_angle)<0.2: 
		            speed_factor_tl_temp = tl_dist/(self.coast_factor*self.tl_dist_thres)
			  if (0 < tl_angle < self.tl_angle_thres*self.coast_factor and tl_dist < self.tl_dist_thres/self.coast_factor) and math.fabs(wp_angle)<0.2:
			    speed_factor_tl_temp = 0

			  if (speed_factor_tl_temp < speed_factor_tl):
			    speed_factor_tl = speed_factor_tl_temp

		    if agent.HasField('pedestrian') and self.stop4P:
		        x_agent = agent.pedestrian.transform.location.x
		        y_agent = agent.pedestrian.transform.location.y
		        p_vector, p_dist = self.get_vec_dist(x_agent, y_agent, loc_x_player, loc_y_player)
		        #p_angle = self.get_angle(p_vector,[ori_x_player,ori_y_player])
			p_angle = self.get_angle(p_vector,wp_vector)
		        #print ('Pedestrian: ', p_vector, p_dist, p_angle)
			if (math.fabs(p_angle) < self.p_angle_thres/self.coast_factor and p_dist < self.p_dist_thres*self.coast_factor) or (0 < p_angle < self.p_angle_thres and p_dist < self.p_dist_thres): 
		          speed_factor_p_temp = p_dist/(self.coast_factor*self.p_dist_thres)
			if (math.fabs(p_angle) < self.p_angle_thres*self.coast_factor and p_dist < self.p_dist_thres/self.coast_factor):
			  speed_factor_p_temp = 0

			if (speed_factor_p_temp < speed_factor_p):
			  speed_factor_p = speed_factor_p_temp

	 	    if agent.HasField('vehicle') and self.stop4V:
		        x_agent = agent.vehicle.transform.location.x
		        y_agent = agent.vehicle.transform.location.y
		        v_vector, v_dist = self.get_vec_dist(x_agent, y_agent, loc_x_player, loc_y_player)
		        #v_angle = self.get_angle(v_vector,[ori_x_player,ori_y_player])
		        v_angle = self.get_angle(v_vector,wp_vector)
		        #print ('Vehicle: ', v_vector, v_dist, v_angle)
 		        print (v_angle,self.v_angle_thres,self.coast_factor)
			if (-0.5*self.v_angle_thres/self.coast_factor < v_angle < self.v_angle_thres/self.coast_factor and v_dist < self.v_dist_thres*self.coast_factor) or (-0.5*self.v_angle_thres/self.coast_factor < v_angle < self.v_angle_thres and v_dist < self.v_dist_thres): 
		          speed_factor_v_temp = v_dist/(self.coast_factor*self.v_dist_thres)
			if (-0.5*self.v_angle_thres*self.coast_factor < v_angle < self.v_angle_thres*self.coast_factor and v_dist < self.v_dist_thres/self.coast_factor):
			  speed_factor_v_temp = 0

			if (speed_factor_v_temp < speed_factor_v):
			  speed_factor_v = speed_factor_v_temp

  		    self.speed_factor = min(speed_factor_tl,speed_factor_p,speed_factor_v)
     		
		self.current_speed = player.forward_speed
		# We should run some state machine around here 
		control = self.controller(wp_angle,wp_mag,wp_angle_speed)

		return control

	def get_active_wps (self):
		return self.wp, self.wp_speed

