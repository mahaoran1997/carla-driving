
class CarlaBase(object):


	def __init__(self,driver_conf):


		Driver.__init__(self)
		self._straight_button = False
		self._left_button = False
		self._right_button = False
		self._recording= False

		self._augment_left_right = driver_conf.augment_left_right

		# load a manager to deal with test data
		self.use_planner = driver_conf.use_planner

		if driver_conf.use_planner:
			self.planner = Planner('drive_interfaces/carla_interface/' + driver_conf.city_name  + '.txt',\
			'drive_interfaces/carla_interface/' + driver_conf.city_name + '.png')

		self._host = driver_conf.host
		self._port = driver_conf.port
		self._config_path = driver_conf.carla_config
		self._resolution = driver_conf.resolution

		self._rear = False

	def start_carla(self):

		self.carla =Carla(self._host,self._port,self._config_path)


		self.carla.startAgent()
	

	


		
		self.positions= self.carla.requestNewEpisode()

		self._current_goal = random.randint(0,len(self.positions))

		self.carla.newEpisode(random.randint(0,len(self.positions)))
		self._dist_to_activate = random.randint(100,500)
		pygame.joystick.init()

		#except:
		#	print 'ERROR: Failed to connect to DrivingServer'
		#else:
		#	print 'Successfully connected to DrivingServer'

		# Now start the joystick, the actual controller 

		
		joystick_count = pygame.joystick.get_count()
		if joystick_count >1:
			print "Please Connect Just One Joystick"
			raise 



		self.joystick = pygame.joystick.Joystick(0)
		self.joystick.init()


	def compute_action(self,sensor,speed):


		""" Get Steering """

		steering_axis = self.joystick.get_axis(0)

		acc_axis = self.joystick.get_axis(2)

		brake_axis = self.joystick.get_axis(3)

		if( self.joystick.get_button( 3 )):

			self._rear =True
		if( self.joystick.get_button( 2 )):
			self._rear=False


		control = Control()
		control.steer = steering_axis
		control.gas = -(acc_axis -1)/2.0
		control.brake = -(brake_axis -1)/2.0
		if control.brake < 0.001:
			control.brake = 0.0
		
			
		control.hand_brake = 0
		control.reverse = self._rear


		if self._augment_left_right: # If augment data, we generate copies of steering for left and right
			control_left = copy.deepcopy(control)

			control_left.steer = self._adjust_steering(control_left.steer,15.0,speed) # The angles are inverse.
			control_right = copy.deepcopy(control)

			control_right.steer = self._adjust_steering(control_right.steer,-15.0,speed)

			return [control_left,control,control_right]

		else:
			return [control]



	def get_sensor_data(self,goal_pos=None,goal_ori=None):
		message = self.carla.getReward()
		data = message[0]
		images = message[2]

		pos = [data.player_x,data.player_y,22 ]
		ori = [data.ori_x,data.ori_y,data.ori_z ]
		
		if self.use_planner:

			if sldist([data.player_x,data.player_y],[self.positions[self._current_goal][0],self.positions[self._current_goal][1]]) < self._dist_to_activate:

				self._current_goal = random.randint(0,len(self.positions))
				self._dist_to_activate = random.randint(100,500)

			direction = self.planner.get_next_command(pos,ori,[self.positions[self._current_goal][0],self.positions[self._current_goal][1],22],(1,0,0))
			
		else:
			direction = 2.0

		Reward.direction = property(lambda self: direction)

		return data,images

	
	def act(self,action):


		self.carla.sendCommand(action)

