import glob
import os


#
# This is a config file, with three parts: Input, Training, Output, which are then joined in Main
#


class configMain:
    def __init__(self):
        # this is used for data balancing. Labels are balanced first, and then for each label group
        # the [-1,1] interval is split into so many equal intervals
        # when we need to sample a mini-batch, we sample bins and then sample inside the bins


        self.steering_bins_perc = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
        self.number_steering_bins = len(self.steering_bins_perc)
        self.batch_size = self.number_steering_bins * 15
        self.batch_size_val = self.number_steering_bins * 15
        self.number_images_val = self.batch_size_val  # Number of images used in a validation Section - Default: 20*

        # self.input_size = (227,227,3)
        # self.manager_name = 'control_speed'
        # y , x
        self.image_size = (88, 200, 3)
        self.network_input_size = (88, 200, 3)
        self.variable_names = ['Steer', 'Gas', 'Brake', 'Hand_B', 'Reverse', 'Steer_N', 'Gas_N', 'Brake_N', 'Pos_X',
                               'Pos_Y', 'Speed', \
                               'C_Gen', 'C_Ped', 'C_Car', 'Road_I', 'Side_I', 'Acc_x', 'Acc_y', 'Acc_z', 'Plat_Ts',
                               'Game_Ts', 'Ori_X', 'Ori_Y', \
                               'Ori_Z', 'Control', 'Camera', 'Angle', 'wp1_x', 'wp1_y', 'wp2_x', 'wp2_y', 'wp1_angle',
                               'wp1_mag', 'wp2_angle', 'wp2_mag']
        # _N is noise, Yaw_S is angular speed


        self.sensor_names = ['rgb']
        self.sensors_size = [(88, 200, 3)]
        self.sensors_normalize = [True]

        # CHANGE THIS FOR A DICTIONARY FOR GOD SAKE
        self.targets_names = ['wp1_angle', 'wp2_angle', 'Steer', 'Gas', 'Brake', 'Speed']
        self.targets_sizes = [1, 1, 1, 1, 1, 1]

        self.inputs_names = ['Control', 'Speed']
        self.inputs_sizes = [4, 1]

        # if there is branching, this is used to build the network. Names should be same as targets
        # currently the ["Steer"]x4 should not be changed
        self.branch_config = [['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"],
                              ['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"], \
                              ['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"],
                              ['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"], ["Speed"]]

        # a list of keep_prob corresponding to the list of layers:
        # 8 conv layers, 2 img FC layer, 2 speed FC layers, 1 joint FC layer, 5 branches X 2 FC layers each
        # This is error prone..... GOES TO THE NETWORK FILE DIRECTLY
        self.dropout = [0.8] * 8 + [0.5] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * len(self.branch_config)

        self.restore = True  # This is if you want to restore a saved model

        self.models_path = os.path.join('models', os.path.basename(__file__).split('.')[0])
        self.train_path_write = os.path.join(self.models_path, 'train')
        self.val_path_write = os.path.join(self.models_path, 'val')
        self.test_path_write = os.path.join(self.models_path, 'test')

        self.number_iterations = 501000  # 500k

        self.pre_train_experiment = None
        # Control the execution of simulation testing during training
        self.perform_simulation_test = False
        self.output_is_on = True
        # self.extra_augment_factor = 6.0
        self.segmentation_model = '/export/vcl-nfs2/shared/matthia1/Github/carla_chauffeur/models/erfnet_small_cityscapes'
        self.segmentation_model_name = "ErfNet_Small"


class configInput(configMain):
    def __init__(self, path='/'):
        configMain.__init__(self)

        """
        st = lambda aug: iaa.Sometimes(0.2, aug)
        oc = lambda aug: iaa.Sometimes(0.1, aug)
        rl = lambda aug: iaa.Sometimes(0.04, aug)
        self.augment = [iaa.Sequential([


            rl(iaa.GaussianBlur((0, 1.3))), # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)), # add gaussian noise to images
            rl(iaa.Dropout((0.0, 0.10), per_channel=0.5)), # randomly remove up to X% of the pixels
            oc(iaa.Add((-20, 20), per_channel=0.5)), # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.25, 2.5), per_channel=0.2)), # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))), # put grayscale


            ],
            random_order=True # do all of the above in random order
        ), iaa.Sequential([ # AUGMENTATION Labels

            rl(iaa.Dropout((0.0, 0.03))), # randomly remove up to X% of the pixels

            rl(iaa.CoarseDropout((0.0, 0.03), size_percent=(0.2, 0.3))), # randomly remove up to X% of the pixels

            ],
            random_order=True # do all of the above in random order
        )
        ]
        """
        self.augment = [None]

        # there are files with data, 200 images each, and here we select which ones to use

        self.dataset_name = 'Carla'

        train_path = os.path.join(path, 'RC28_wpz_M')
        val_path = os.path.join(path, 'RC025Val')
        print(train_path)

        self.train_db_path = [os.path.join(train_path, f) for f in glob.glob1(train_path, "data_*.h5")]
        self.val_db_path = [os.path.join(val_path, f) for f in glob.glob1(val_path, "data_*.h5")]

        # Speed Divide Factor

        # TODO: FOr now is hardcooded, but eventually we should be able to calculate this from data at the loading time.
        self.speed_factor = 40.0  # In KM/H FOR GTA it should be maximun 30.0

        # The division is made by three diferent data kinds
        # in every mini-batch there will be equal number of samples with labels from each group
        # e.g. for [[0,1],[2]] there will be 50% samples with labels 0 and 1, and 50% samples with label 2
        self.labels_per_division = [[0, 2, 5], [3], [4]]

        self.dataset_names = ['targets']

        self.queue_capacity = 20 * self.batch_size

    # TODO NOT IMPLEMENTED Felipe: True/False switches to turn data balancing on or off


class configTrain(configMain):
    def __init__(self):
        configMain.__init__(self)

        self.loss_function = 'mse_branched'  # Chose between: mse_branched, mse_branched_ladd
        self.control_mode = 'single_branch_wp'
        self.learning_rate = 0.0002  # First
        self.seg_network_erfnet_one_hot = True
        self.train_segmentation = True
        # self.finetune_segmentation = True
        self.number_of_labels = 2
        self.restore_seg_test = False
        self.training_schedule = [[50000, 0.5], [100000, 0.5 * 0.5], [150000, 0.5 * 0.5 * 0.5],
                                  [200000, 0.5 * 0.5 * 0.5 * 0.5],
                                  [250000, 0.5 * 0.5 * 0.5 * 0.5 * 0.5]]  # Number of iterations, multiplying factor
        self.lambda_l2 = 1e-3  # Not used
        self.branch_loss_weight = [0.95, 0.95, 0.95, 0.95, 0.05]
        self.variable_weight = {'wp1_angle': 0.3, 'wp2_angle': 0.3, 'Steer': 0.1, 'Gas': 0.2, 'Brake': 0.1,
                                'Speed': 1.0}
        self.network_name = 'chauffeurNet_deeper'
        self.lambda_tolerance = 5
        self.is_training = True
        self.selected_gpu = "0"
        self.state_output_size = (0)


class configOutput(configMain):
    def __init__(self):
        configMain.__init__(self)

        self.print_interval = 100
        self.summary_writing_period = 100
        self.validation_period = 1000  # I consider validation as an output thing since it does not directly affects the training in general
        self.feature_save_interval = 100
        self.use_curses = False  # If we want to use curses library for a cutter print

        self.targets_to_print = ['wp1_angle', 'wp2_angle', 'Steer', 'Gas',
                                 'Brake']  # Also prints the error. Maybe Energy
        self.selected_branch = 0  # for the branches that have steering we also select the branch
        self.inputs_to_print = ['Speed']

        """ Feature Visualization Part """

    # TODO : self.histograms_list=[]	self.features_to_draw=  self.weights_to_draw=


class configTest(configMain):
    def __init__(self):
        configMain.__init__(self)

        self.interface_name = 'Carla'

        self.driver_config = "9cam_agent_carla_test_rc"  # "3cam_carla_drive_config"

    # This is the carla configuration related stuff
