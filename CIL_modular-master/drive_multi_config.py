import sys, os, time, threading
from configparser import ConfigParser
from drive import drive
from multiprocessing import Process

sys.path.append('drive_interfaces/configuration')


def config_naming(tag, config, weather):
    out = tag + "_" + \
          config[0] + "=" + config[1] + "_" + \
          "WeatherId=" + str(weather).zfill(2)
    return out


def config_change_attrs(src, dst, new_attrs):
    config = ConfigParser()
    config.optionxform = str
    config.read(src)
    for one in new_attrs:
        config.set(one[0], one[1], one[2])
    with open(dst, "w") as f:
        config.write(f)

def process_collect(list_of_configs, port, gpu,
                    tag, generated_config_cache_path, template_path, driver_config, TownName):
    print("port is ", port, "!!!!!!!!!!!!!!!!!!")
    port = int(port)
    count = 5  # to flag that initially we need to start the server
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for config, weather in list_of_configs:
        # generate this config
        this_name = config_naming(tag, config, weather)
        config_fname = os.path.join(generated_config_cache_path, this_name + ".ini")

        config_change_attrs(template_path, config_fname,
                            [("CARLA/Sensor", config[0], config[1]),
                             ("CARLA/LevelSettings", "WeatherId", str(weather))])

        driver_conf_module = __import__(driver_config)
        driver_conf = driver_conf_module.configDrive()
        driver_conf.carla_config = config_fname
        driver_conf.weather = str(weather)
        driver_conf.port = port

        # experiment_name & memory not used for human

        while True:
            count += 1
            if count >= 5:
                count = 0
                cmd = ['bash', '-c',
                       " '/scratch/haoran/carla/CarlaUE4.sh /Game/Maps/%s  -carla-server -benchmark -fps=5 -world-port=%d -carla-settings=../carla/Example.CarlaSettings.ini' " % (TownName, port)]
                print(" ".join(cmd))
                print("before spawnling")
                t = threading.Thread(target=lambda: os.system(" ".join(cmd)))
                t.start()

                time.sleep(15)

            if drive("", driver_conf, this_name, 0):
                count = 0
                break
            time.sleep(1)

        print("finished one setting, sleep for 3 seconds")
        time.sleep(3)


if __name__ == "__main__":
    collect_all = False

    driver_config = "9cam_agent_carla_acquire_rc_batch"
    if collect_all:
        driver_config = "9cam_agent_carla_acquire_rc_batch_allsensors"

    TownName = "Town01"
    if collect_all:
        TownName = "Town01"

    generated_config_cache_path = "./drive_interfaces/carla/auto_gen_configs/"
    tag = "default"
    # TODO: tune those base config
    template_path = "./drive_interfaces/carla/yang_template.ini"
    if collect_all:
        template_path = "./drive_interfaces/carla/yang_template_all.ini"
    template_path = "./drive_interfaces/carla/yang_template_3cams.ini"

    # TODO: tune those numbers
    # (propertyName, potential value)
    configs = [("RotationPitch", "0"), # This is the default one
               ("RotationPitch", "5"),
               ("RotationPitch", "-5"),
               ("ImageSizeX", "600"),
               ("ImageSizeX", "700"),
               ("ImageSizeX", "900"),
               ("ImageSizeX", "1000"),
               ("ImageSizeX", "1100"),
               ("ImageSizeX", "1200"),
               ("PositionZ", "0.5"),
               ("PositionZ", "1.5")]
    weather_range = range(1, 14)
    # all the input params ends here
    configs = [("RotationPitch", "0")]
    weather_range = range(14)
    # a simple test config ends here
    configs = [("RotationPitch", "0"),  # This is the default one
               ("RotationPitch", "5"),
               ("RotationPitch", "-5"),
               ("ImageSizeX", "700"),
               ("ImageSizeX", "800"),
               ("PositionZ", "1.4"),
               ("PositionZ", "1.8")]
    weather_range = range(1, 14)

    if collect_all:
        configs = [("RotationPitch", "0") ]

    # a test case
    configs = [("RotationPitch", "0")]
    weather_range = [1]
    # end of test case

    # the noiser setting
    configs = [("RotationPitch", "0"),  # This is the default one
               ("RotationPitch", "5"),
               ("RotationPitch", "-5"),
               ("ImageSizeX", "700"),
               ("ImageSizeX", "800"),
               ("PositionZ", "1.4"),
               ("PositionZ", "1.8")]
    weather_range = range(1, 15)

    # in total there are 7*14 = 100 configs, each of the 200 h5 file has size of 33M, i.e. 30 h5 = 1G
    # Thus we aim to collect 100 hours of training, that is 400G, so each config has quota of 3G, which is 100 files
    # an initial config ends here



    #available_gpus = [0, 2, 4, 5, 6]
    #num_processes = len(available_gpus) * 2
    available_gpus = [0]
    num_processes = 4

    list_of_configs = [[] for i in range(num_processes)]

    counter = 0
    for config in configs:
        for weather in weather_range:
            list_of_configs[counter % num_processes].append((config, weather))
            counter += 1

    ps=[]
    for i in range(num_processes):
        p = Process(target=process_collect, args=(list_of_configs[i],
                                                  4000+i*3,
                                                  available_gpus[i % len(available_gpus)],
                                                  tag,
                                                  generated_config_cache_path,
                                                  template_path,
                                                  driver_config,
                                                  TownName))
        p.start()
        print("after starts!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(5)
        ps.append(p)
        print("finsished starting process ", i)

    for i in range(num_processes):
        ps[i].join()
