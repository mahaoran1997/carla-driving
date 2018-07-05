from carla.sensor import Camera

class configDrive:

  # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
  # TODO: kind of change this to be CarlaSettings Based ?
  def __init__(self):

    #self.experiment_name =''
    self.carla_config ="./drive_interfaces/carla/iCarla9CamerasW1T.ini"   # The path to carla ini file # TODO: make this class to be able to generate the ini file
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
    self.scale_factor = 1 # NON CARLA SETTINGS PARA M
    self.image_cut =[115,510] # This is made from top to botton
    self.autopilot = True # Data collection parameters
    self.reset_period = 960
    # Figure out a solution for setting specific properties of each interface
    self.use_planner = False
    self.city_name  = 'Town02'
    self.plot_vbp = False # Data Plotting Param

    self.camera_set =[]

    camera = Camera('RGB')
    camera.set(CameraFOV=100)
    camera.set_image_size(800, 600)
    camera.set_position(200, 0, 140)
    camera.set_rotation(-15.0,0,0)
    self.camera_set.append(camera)

    camera = Camera('Labels')
    camera.set(CameraFOV=100,PostProcessing='SemanticSegmentation')
    camera.set_image_size(800, 600)
    camera.set_position(200, 0, 140)
    camera.set_rotation(-15.0,0,0)
    self.camera_set.append(camera)