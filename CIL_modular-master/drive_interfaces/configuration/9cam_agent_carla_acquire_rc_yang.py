class configDrive:
    # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
    # TODO: kind of change this to be CarlaSettings Based ?
    def __init__(self):
        # self.experiment_name =''

        # human or machine
        self.type_of_driver = "Human"

        # resource related
        self.host = "127.0.0.1"
        self.port = 2000
        self.path = "../Desktop/"  # If path is set go for it , if not expect a name set

        # data collection related
        self.city_name = 'carla_1'
        self.carla_config = "./drive_interfaces/carla/rcCarla_9Cams_W1.ini"  # The path to carla ini file # TODO: make this class to be able to generate the ini file
        self.middle_camera = 0  # might be wrong, based on the actual camera?
        # collect method
        self.autopilot = True
        self.use_planner = True
        self.noise = "None"  # NON CARLA SETTINGS PARAM
        self.reset_period = 960
        # Test parameters to be shared between models
        self.timeouts = [200.0]  # 130
        self.weather = "1"
        self.cars = "50"
        self.pedestrians = "100"

        # post processing
        self.image_cut = [200, 550]  # This is made from top to botton
        self.resolution = [200, 88]

        # Visualization related
        self.show_screen = False  # NON CARLA SETTINGS PARAM
        self.aspect_ratio = [3, 1]  # only matters when showing the screen
        self.scale_factor = 1  # NON CARLA SETTINGS PARAM
        self.plot_vbp = False

        # others
        self.interface = "Carla" # always fixed
