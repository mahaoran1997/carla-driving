
class configDrive:

  # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
  # TODO: kind of change this to be CarlaSettings Based ?
  def __init__(self):

    #self.experiment_name =''
    self.carla_config = None #"./drive_interfaces/carla/iCarla9CamerasW1.ini"   # The path to carla ini file # TODO: make this class to be able to generate the ini file
    self.host = "127.0.0.1"
    self.port = 2000
    self.path = "/scratch/haoran/data/carla_collect/noiser_3cam/" # If path is set go for it , if not expect a name set
    self.resolution = [225, 225]
    self.noise = "None" #NON CARLA SETTINGS PARAM
    self.type_of_driver = "Human"
    self.interface = "Carla"
    self.show_screen = False #NON CARLA SETTINGS PARAM
    self.aspect_ratio = [3,1]
    self.middle_camera =0
    self.scale_factor = 1 # NON CARLA SETTINGS PARAM
    self.image_cut =[100, 450, 225, 575] # This is made from top to botton
    self.autopilot = True
    self.reset_period = 960
    # Figure out a solution for setting specific properties of each interface
    self.use_planner = True
    self.city_name  = 'carla_1'
    self.plot_vbp = False
    # Test parameters to be shared between models

    self.timeouts =[130.0]
    self.weather =1
    self.cars = 0
    self.pedestrians =0
