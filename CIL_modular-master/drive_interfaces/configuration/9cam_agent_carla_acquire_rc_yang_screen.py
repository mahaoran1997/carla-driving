class configDrive:
    # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
    # TODO: kind of change this to be CarlaSettings Based ?
    def __init__(self):
        # self.experiment_name =''
        self.carla_config = "./drive_interfaces/carla/rcCarla_9Cams_W1.ini"  # The path to carla ini file # TODO: make this class to be able to generate the ini file
        self.host = "127.0.0.1"
        self.port = 2000
        self.path = "../Desktop/"  # If path is set go for it , if not expect a name set
        self.resolution = [200, 88]
        self.noise = "None"  # NON CARLA SETTINGS PARAM # TODO: experiment with this
        self.type_of_driver = "Human"
        self.interface = "Carla"
        self.show_screen = True  # NON CARLA SETTINGS PARAM
        self.aspect_ratio = [3, 1]  # only matters when showing the screen
        self.middle_camera = 1  # might be wrong, based on the actual camera?
        self.scale_factor = 3  # NON CARLA SETTINGS PARAM
        self.image_cut = [200, 550]  # This is made from top to botton
        self.autopilot = True
        self.reset_period = 960
        # Figure out a solution for setting specific properties of each interface
        self.use_planner = True # TODO: experiment with this
        self.city_name = 'carla_1'
        self.plot_vbp = True
        # Test parameters to be shared between models

        self.timeouts = [200.0]  # 130
        self.weather = "1"
        self.cars = "50"
        self.pedestrians = "100"
