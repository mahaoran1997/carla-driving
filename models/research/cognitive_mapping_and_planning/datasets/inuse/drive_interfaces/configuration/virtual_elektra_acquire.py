

class configDrive:

  # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
  # TODO: kind of change this to be CarlaSettings Based ?
  def __init__(self):

    #self.experiment_name =''
    self.carla_config ="./drive_interfaces/carla/elektra_more_cams.ini"   # The path to carla ini file # TODO: make this class to be able to generate the ini file
    self.host = "127.0.0.1"
    self.port = 2000
    self.path = "../Desktop/" # If path is set go for it , if not expect a name set
    self.resolution = [200,88]
    self.noise = "None" #NON CARLA SETTINGS PARAM
    self.type_of_driver = "Human"
    self.interface = "Carla"
    self.show_screen = True #NON CARLA SETTINGS PARAM
    self.aspect_ratio = [3,1]
    self.middle_camera =0
    self.scale_factor = 1 # NON CARLA SETTINGS PARAM
    self.image_cut =[115,510] # This is made from top to botton
    self.autopilot = False
    self.reset_period = 240
    # Figure out a solution for setting specific properties of each interface
    self.use_planner = False
    self.city_name  = 'carla_1'
    self.plot_vbp = False
    # Test parameters to be shared between models

    self.weather =1
    self.cars = 50
    self.pedestrians =100
