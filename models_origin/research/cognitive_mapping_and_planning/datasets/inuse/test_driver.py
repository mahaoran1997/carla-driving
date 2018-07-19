from datasets.inuse.carla_env import *




def test_driver(experiment_name, drive_config, name=None, memory_use=1.0):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name
    pass
'''
    driver, recorder = get_instance(drive_config, experiment_name, name, memory_use)

    noiser = Noiser(drive_config.noise)

    print 'before starting'
    driver.start()
    first_time = True
    if drive_config.show_screen:
        screen_manager = ScreenManager()
        screen_manager.start_screen(drive_config.resolution, drive_config.aspect_ratio,
                                    drive_config.scale_factor)  # [800,600]

    direction = 2

    iteration = 0
    try:
        while direction != -1:
            capture_time = time.time()
            print (iteration)
            if hasattr(drive_config, 'carla_config'):
                measurements, direction = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]
            else:  ##RC Car
                measurements, images = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]

            # sensor_data = frame2numpy(image,[800,600])

            # Compute now the direction
            if drive_config.show_screen:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done = True  # Flag that we are done so we exit this loop

            recording = driver.get_recording()
            driver.get_reset()
            if hasattr(drive_config, 'carla_config'):
                speed = measurements['PlayerMeasurements'].forward_speed
                # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed,\
                # driver.compute_direction((measurements.transform.location.x,measurements.transform.location.y,22),\
                # (measurements.transform.orientation.x,measurements.transform.orientation.y,measurements.transform.orientation.z))) #rewards.speed
                # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed) #rewards.speed
                actions = driver.compute_action([measurements['BGRA'][drive_config.middle_camera],
                                                 measurements['Labels'][drive_config.middle_camera]],
                                                speed)  # measurements.speed
                action_noisy, drifting_time, will_drift = noiser.compute_noise(actions, speed)

            else:  ##RC Car
                actions = driver.compute_action(images[drive_config.middle_camera], 0)  # measurements.speed
                action_noisy, drifting_time, will_drift = noiser.compute_noise(actions[drive_config.middle_camera])

            # print actions
            if recording:
                if drive_config.interface == "DeepRC":

                    recorder.record(images, measurements, actions, action_noisy)
                else:
                    recorder.record(measurements, actions, action_noisy, direction, driver.get_waypoints())

            if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:

                if drive_config.interface == "DeepRC":
                    image_vbp = driver.compute_perception_activations(images[drive_config.middle_camera], 0)
                else:
                    image_vbp = driver.compute_perception_activations(image, speed)

                screen_manager.plot_camera(image_vbp, [1, 0])

            iteration += 1
            driver.act(action_noisy)

    except:
        traceback.print_exc()

    finally:

        # driver.write_performance_file(path,folder_name,iteration)
        pygame.quit()
'''