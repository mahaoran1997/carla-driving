import glob, os
from imgaug import augmenters as iaa

class configMain:
    def __init__(self):
        self.steering_bins_perc = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
        self.number_steering_bins = len(self.steering_bins_perc)
        self.batch_size = self.number_steering_bins * 6
        self.batch_size_val = self.number_steering_bins * 6
        self.number_images_val = self.batch_size_val  # Number of images used in a validation Section - Default: 20*

        self.image_size = (576, 768, 3)
        # the speed unit has changed, when reading from the h5 file, we change the speed_factor to adjust for this
        self.variable_names = ['Steer', 'Gas', 'Brake', 'Hand_B', 'Reverse',
                               'Steer_N', 'Gas_N', 'Brake_N',
                               'Pos_X', 'Pos_Y', 'Speed',
                               'C_Gen', 'C_Ped', 'C_Car', 'Road_I', 'Side_I', 'Acc_x', 'Acc_y', 'Acc_z',
                               'Plat_Ts', 'Game_Ts', 'Ori_X', 'Ori_Y', 'Ori_Z', 'Control', 'Camera', 'Angle',
                               'wp1_x', 'wp1_y', 'wp2_x', 'wp2_y', 'wp1_angle', 'wp1_mag', 'wp2_angle', 'wp2_mag']

        self.sensor_names = ['CameraMiddle']

        self.targets_names = ['Steer', 'Gas', 'Brake', 'Speed']
        self.targets_sizes = [1, 1, 1, 1]

        self.inputs_names = ['Control', 'Speed']
        self.inputs_sizes = [4, 1]

        # if there is branching, this is used to build the network. Names should be same as targets
        # currently the ["Steer"]x4 should not be changed
        self.branch_config = [["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"], ["Speed"]]

        # a list of keep_prob corresponding to the list of layers:
        # 7 conv layers, 2 img FC layer, 2 speed FC layers, 1 joint FC layer, 5 branches X 2 FC layers each
        #                     3072*512 512**2              640*512  512*256 256**2
        self.dropout = [.95] * 7 + [0.95] * 2 + [.95] * 2 + [0.95] * 1 + [0.95, .95] * len(self.branch_config)

        self.models_path = os.path.join('models', os.path.basename(__file__).split('.')[0])
        self.train_path_write = os.path.join(self.models_path, 'train')
        self.val_path_write = os.path.join(self.models_path, 'val')

        self.number_iterations = 501000  # 500k
        self.perform_simulation_test = False
        self.output_is_on = True

        # disable the segmentation model
        self.segmentation_model = None
        #self.segmentation_model_name = "ErfNet_Small"

        # perception module related
        self.use_perception_stack = True
        self.perception_gpus = [1]
        self.perception_paths = "path_jormungandr_newseg"
        self.perception_batch_sizes = {"det_COCO": 3, "det_TL": 3, "seg": 4, "depth": 4, "det_TS": -1}
        self.perception_num_replicates = {"det_COCO": -1, "det_TL": -1, "seg": 4, "depth": -1, "det_TS": -1}
        # debug
        #self.perception_num_replicates = {"det_COCO": -1, "det_TL": -1, "seg": -1, "depth": 1, "det_TS": -1}
        if self.use_perception_stack:
            self.feature_input_size = (39, 52, 295)  # hardcoded for now
            self.image_as_float = [False]
            self.sensors_normalize = [False]
            self.perception_initialization_sleep=15
            # debug
            self.feature_input_size = (39, 52, 54)
        else:
            self.feature_input_size = self.image_size

        self.optimizer = "adam" # or "adam"


class configInput(configMain):
    def __init__(self):
        configMain.__init__(self)

        st = lambda aug: iaa.Sometimes(0.2, aug)
        oc = lambda aug: iaa.Sometimes(0.1, aug)
        rl = lambda aug: iaa.Sometimes(0.05, aug)

        self.augment = [iaa.Sequential([
            rl(iaa.GaussianBlur((0, 1.3))),  # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
            rl(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
            oc(iaa.Add((-20, 20), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.25, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))),  # put grayscale
            ],
            random_order=True  # do all of the above in random order
        )]

        all_files = glob.glob("/data/yang/code/aws/scratch/carla_collect/noiser_direction/*/data_*.h5")
        self.val_db_path = glob.glob("/data/yang/code/aws/scratch/carla_collect/noiser_direction/*WeatherId=1[34]/data_*.h5")
        self.train_db_path = list(set(all_files) - set(self.val_db_path))

        self.speed_factor = 40.0  # In KM/H

        # The division is made by three diferent data kinds
        # in every mini-batch there will be equal number of samples with labels from each group
        # e.g. for [[0,1],[2]] there will be 50% samples with labels 0 and 1, and 50% samples with label 2
        self.labels_per_division = [[0, 2, 5], [3], [4]]
        self.dataset_names = ['targets']
        self.queue_capacity = 5 # now measured in how many batches

    # TODO NOT IMPLEMENTED Felipe: True/False switches to turn data balancing on or off


class configTrain(configMain):
    def __init__(self):
        configMain.__init__(self)

        self.loss_function = 'mse_branched'  # Chose between: mse_branched, mse_branched_ladd
        self.control_mode = 'single_branch'
        # TODO: tune it
        self.learning_rate = 1e-4
        # use the default segmentation network
        #self.seg_network_erfnet_one_hot = True  # comment this line out to use the standard network

        # TODO: tune it
        factor = 0.3333
        # Number of iterations, multiplying factor
        self.training_schedule = [[50000, factor**1], [75000, factor**2], [100000, factor**3]]

        self.branch_loss_weight = [0.95, 0.95, 0.95, 0.95, 0.05]
        self.variable_weight = {'Steer': 0.3, 'Gas': 0.2, 'Brake': 0.1, 'Speed': 1.0}
        self.network_name = 'yang_39_52_295_v2'
        self.is_training = True


class configOutput(configMain):
    def __init__(self):
        configMain.__init__(self)
        self.print_interval = 10 # how often in output manager to print
        self.summary_writing_period = 100 # for output manager
        self.validation_period = 1000  # For output manager, I consider validation as an output thing since it does not directly affects the training in general



# The class below is deprecated, but save it for now, because we have to migrate the testing process to the new benchmark.
class configTest(configMain):
    def __init__(self):
        configMain.__init__(self)
        self.driver_config = "9cam_agent_carla_test_rc"  # only used, when calling the drive() function
