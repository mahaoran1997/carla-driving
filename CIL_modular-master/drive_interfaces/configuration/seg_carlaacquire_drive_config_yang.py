class configDrive:
    # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
    # TODO: kind of change this to be CarlaSettings Based ?
    def __init__(self):
        # self.experiment_name =''
        self.carla_config = "./drive_interfaces/carla_interface/CarlaSettingsSegmentation.ini"  # The path to carla ini file # TODO: make this class to be able to generate the ini file
        self.host = "127.0.0.1"
        self.port = 2000
        self.path = "../Desktop/"  # If path is set go for it , if not expect a name set
        self.resolution = [300, 200]  # [200,88],[800,600]
        self.noise = "None"  # NON CARLA SETTINGS PARAM
        self.type_of_driver = "Human"
        self.interface = "Carla"
        self.show_screen = False  # NON CARLA SETTINGS PARAM
        self.number_screens = 1  # if show_screen=False, then number_screens does not have an effect
        self.middle_camera = 0  # measurements[index] that is the middle camera
        self.scale_factor = 1  # NON CARLA SETTINGS PARAM, only used when show_screen
        # self.image_cut =[115,510],[170,518] # This is made from top to botton
        self.image_cut = [0, 600]
        self.augment_left_right = False  # If true also generates steer for turning left and turning right
        self.camera_angle = 0  # when using the augment above, what degree to set for the left and right camera
        self.autopilot = True
        self.reset_period = 10000  # reset after autopilot drive for this amount of time
        # Figure out a solution for setting specific properties of each interface
        self.use_planner = False  # direction is a constant, 2.0, Not sure what it is used for.
        self.city_name = 'carla_1'  # carla_0, carla_1
        self.plot_vbp = False  # if true, visualize perception activations
        # Test parameters to be shared between models

        #### POSITIONS FOR CARLA 1 #####
        # self.episodes_positions =[
        # [105,29],[140,134]#,[65,18],[21,16],[97,64],[121,85],[30,41],[18,107],[69,45],[102,95],\
        # [104,137],[72,95],[142,96],[7,115],[7,104],[34,70],[132,27],[24,44],[1,42],[37,32]\
        # ]
        self.episodes_positions = [
            [53, 76], [42, 13]  # ,[79,19],[2,29],[70,73],[46,67],[51,81],[77,68],[29,76],[49,63],\
            # [24,64],[19,50],[65,56],[54,43],[75,40],[58,25]
        ]

        self.timeouts = [130.0] * len(self.episodes_positions)
        self.weathers = [1] * len(self.episodes_positions)
        self.number_of_cars = [15] * len(self.episodes_positions)
        self.number_of_pedestrians = [50] * len(self.episodes_positions)
