

class configDrive:

  # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
  # TODO: kind of change this to be CarlaSettings Based ?
  def __init__(self):

    #self.experiment_name =''
    self.carla_config ="./drive_interfaces/carla/rcCarla_9Cams_W1.ini"   # The path to carla ini file # TODO: make this class to be able to generate the ini file
    self.host = "127.0.0.1"
    self.port = 2000
    self.path = "path/" # If path is set go for it , if not expect a name set
    self.resolution = [225,225] #[200, 88]?
    self.noise = "None" #NON CARLA SETTINGS PARAM
    self.type_of_driver = "Human"
    self.interface = "Carla"
    self.aspect_ratio = [3,1]
    self.middle_camera =0
    self.scale_factor = 1 # NON CARLA SETTINGS PARAM
    self.image_cut = [100, 450, 225, 575]#[200,550] # This is made from top to bottom
    self.reset_period = 960
    # Figure out a solution for setting specific properties of each interface
    self.city_name  = 'carla_1'
    self.plot_vbp = False
    # Test parameters to be shared between models

    self.timeouts =[500.0] #130
    self.weather =1
    #self.cars = 50
    #self.pedestrians =100
    self.typ = 'rgb'
    self.map_scales = [0.03125, 0.0625, 0.125]
    self.map_crop_sizes = [16, 16, 16]
    self.n_ori = 4

    self.reward_at_goal = 1.0
    self.reward_time_penalty = 0.1 
