import sys

sys.path.append('test')
sys.path.append('configuration')
sys.path.append('input')
sys.path.append('utils')
sys.path.append('input/spliter')

import argparse

import os

import logging


def parse_drive_arguments(args, driver_conf):
    # Carla Config
    if args.carla_config is not None:
        driver_conf.carla_config = args.carla_config

    if args.host is not None:
        driver_conf.host = args.host

    if args.port is not None:
        driver_conf.port = int(args.port)

    if args.path is not None:
        driver_conf.path = args.path

    if args.noise is not None:
        driver_conf.noise = args.noise
    if args.driver is not None:
        driver_conf.type_of_driver = args.driver
    if args.interface is not None:
        driver_conf.interface = args.interface
    if args.number_screens is not None:
        driver_conf.number_screens = args.number_screens
    if args.scale_factor is not None:
        driver_conf.scale_factor = args.scale_factor

    if args.resolution is not None:
        res_string = args.resolution.split(',')
        resolution = []
        resolution.append(int(res_string[0]))
        resolution.append(int(res_string[1]))
        driver_conf.resolution = resolution

    if args.image_cut is not None:
        cut_string = args.image_cut.split(',')
        image_cut = []
        image_cut.append(int(cut_string[0]))
        image_cut.append(int(cut_string[1]))
        driver_conf.image_cut = image_cut

    return driver_conf



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DataCollection')
    # General
    parser.add_argument('mode', metavar='mode', default='train', type=str, help='what kind of mode you are running')

    parser.add_argument('-g', '--gpu', type=str, default="0", help='GPU NUMBER')
    parser.add_argument('-lg', '--log', help="activate the log file", action="store_true")
    parser.add_argument('-db', '--debug', help="put the log file to screen", action="store_true")
    parser.add_argument('-e', '--experiment-name',
                        help="The experiment name (NAME.py file should be in configuration folder, and the results will be saved to models/NAME)",
                        default="")
    parser.add_argument('-pt', '--path', type=str, default="path/", help='Path to Store or read outputs') #"../Desktop/"


    # Train

    parser.add_argument('-m', '--memory', default=0.2, help='The amount of memory this process is going to use')


    # VALIDATE

    parser.add_argument('-a', '--angle', default=None, help='The cameras to use for validation')


    # Drive

    parser.add_argument('-dc', '--driver-config', type=str, help="The configuration of the driving file")
    parser.add_argument('-cc', '--carla-config', help="Carla config file used for driving")
    parser.add_argument('-l', '--host', type=str, help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default="2000", help='The port where Any server to be connected is running')

    parser.add_argument('-w', '--weather_used', default=1, help='The weather used on test drive')
    parser.add_argument('-nm', '--name', type=str, default="Felipe", help='Name of the person who is going to drive')
    parser.add_argument('-sc', '--show_screen', default=True, action="store_true",
                        help='If we are showing the screen of the player')
    parser.add_argument('-res', '--resolution', help='If we are showing the screen of the player')
    parser.add_argument('-n', '--noise', help='Set the types of noise:  Spike or None')
    parser.add_argument('--driver', help='Select who is driving, a human or a machine')
    parser.add_argument('-in', '--interface', help='The environment being used as interface')
    parser.add_argument('-cy', '--city',default='Town02', type=str, help='select the graph from the city being used')
    parser.add_argument('-nc', '--number_screens', help='Set The number of screens that are being ploted')
    parser.add_argument('-sf', '--scale_factor', help='Set the scale of the ploted screens')
    parser.add_argument('-up', '--use_planner', help='Check if the planner is going to be used')
    parser.add_argument('-imc', '--image_cut', help='Set the positions where the image is cut')

    args = parser.parse_args()
    know_args = parser.parse_known_args()

    if args.log or args.debug:
        LOG_FILENAME = 'log_manual_control.log'
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
        if args.debug:  # set of functions to put the logging to screen

            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            root.addHandler(ch)

    try:
        if args.mode == 'drive':
            from drive import drive
            driver_conf_module = __import__(args.driver_config)
            driver_conf = driver_conf_module.configDrive()

            driver_conf = parse_drive_arguments(args, driver_conf)

            drive(driver_conf, args.name, float(args.memory))

        elif args.mode == 'test_drive':
            pass

        else:
            pass

    except KeyboardInterrupt:
        os._exit(1)
        exitapp = True
        raise



'''         from drive import drive

            driver_conf_module = __import__(args.driver_config)
            driver_conf = driver_conf_module.configDrive()

            driver_conf = parse_drive_arguments(args, driver_conf)

            drive(args.experiment_name, driver_conf, args.name, float(args.memory))
'''
'''         from test_drive import test_drive


            driver_conf_module = __import__(args.driver_config)
            driver_conf = driver_conf_module.configDrive()

            driver_conf = parse_drive_arguments(args, driver_conf)

            test_drive(args.host, int(args.port), args.experiment_name, driver_conf,
                       args.city, int(args.weather_used), float(args.memory))
'''