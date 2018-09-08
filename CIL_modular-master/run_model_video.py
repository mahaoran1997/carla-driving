import cv2, sys, os
sys.path.append("drive_interfaces/carla/comercial_cars")
from carla_machine import *
from subprocess import call

def loop_over_video(path, func, output_path, temp_down_factor=10, batch_size=1):
    # from a video, use cv2 to read each frame

    # reading from a video
    cap = cv2.VideoCapture(path)

    i = 0
    batch_frames = []
    video_init = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if i % temp_down_factor:
            i += 1
            continue
        print(i)
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            # frame is the one
            print("calling loop function...")
            frame_seq = func(batch_frames)
            print("calling loop function finished")
            if not video_init:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                video = cv2.VideoWriter(output_path, fourcc, max(30 // temp_down_factor, 1),
                                        (frame_seq[0].shape[1], frame_seq[0].shape[0]))
                print("in test_video.loop_over_video, loop function output size:", frame_seq[0].shape)
                video_init = True
            for frame in frame_seq:
                video.write(frame)
            batch_frames = []
        i += 1

    cap.release()
    video.release()

def model_function(batch_frames, vehicle_real_speed_kmh, direction, driving_model):
    out = []
    for frame in batch_frames:
        frame = frame[:,:,:] # the frames comes in bgr
        # TODO: right now, this program only support eval on a single video stream, in the future it will support multiple cameras
        sensors = [frame]
        control, vis = driving_model.compute_action(sensors, vehicle_real_speed_kmh, direction,
                                                    save_image_to_disk=False, return_vis=True)
        out.append(vis[:,:,::-1])

    return out

def get_driver_config():
    driver_conf = lambda: None  # an object that could add attributes dynamically
    driver_conf.image_cut = [0, 100000]
    driver_conf.host = None
    driver_conf.port = None
    driver_conf.use_planner = False  # fixed
    driver_conf.carla_config = None  # This is not used by CarlaMachine but it's required
    return driver_conf


if __name__ == "__main__":
    # start of configurable params
    path = "/scratch/yang/aws_data/mkz/mkz2/inverted_compress.avi"
    exp_id = "mm45_v4_perception_straight3constantaug_lessdrop_yangv2net_segonly"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    real_speed_kmh = 0.0
    direction = 5.0
    temp_down = 1
    # end of configurable params

    out_prefix = path.split(".")[0] + "_" + exp_id + "_speed" + str(real_speed_kmh) + "_direction" + str(direction)
    output_path = out_prefix + ".avi"
    output_compressed = out_prefix + "_264" + ".mp4"

    driving_model = CarlaMachine("0", exp_id, get_driver_config(), 0.1)
    func = lambda batch_frames: model_function(batch_frames, real_speed_kmh, direction, driving_model)

    loop_over_video(path, func, output_path, temp_down, 1)

    cmd = "ffmpeg -i %s -vcodec libx264 %s" % (output_path, output_compressed)
    call(cmd, shell=True)
