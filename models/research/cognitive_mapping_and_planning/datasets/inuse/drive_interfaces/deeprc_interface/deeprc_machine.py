
import numpy as np
import cv2
import mavros_msgs as msgs
import scipy
#/mavros/rc/override - Override RC inputs
from mavros_msgs.msg import OverrideRCIn

#/mavros/set_stream_rate - Send stream rate request to FCU
from mavros_msgs.srv import StreamRate 

#/mavros/set_mode - Set FCU operation mode (http://wiki.ros.org/mavros/CustomModes) 
from mavros_msgs.srv import SetMode


#/mavros/global_position/gp_vel - Velocity fused by FCU
#/mavros/global_position/raw/gps_vel - Velocity output from the GPS device
#/mavros/local_position/velocity - Velocity data from FCU
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

#/mavros/global_position/rel_alt - Relative altitude
#/mavros/global_position/compass_hdg - Compass heading in degrees
from std_msgs.msg import Float64
import math
import copy
from driver import *
import logging
import tensorflow as tf
from training_manager import TrainManager
import machine_output_functions
import os
from deeprc_callbacks import *
import time


from drawing_tools import *
print(cv2.__version__)

def restore_session(sess,saver,models_path):

  ckpt = 0
  if not os.path.exists(models_path):
    os.mkdir( models_path)
  
  ckpt = tf.train.get_checkpoint_state(models_path)
  if ckpt:
    print 'Restoring from ',ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
  else:
    ckpt = 0

  return ckpt


def load_system(config):

  config.batch_size =1
  config.is_training=False

  training_manager= TrainManager(config,None)

  if hasattr(config, 'rgb_seg_network_one_hot'):
    training_manager.build_rgb_seg_network_one_hot()
    print("Bulding: rgb_seg_network_one_hot")

  else:
    if hasattr(config, 'seg_network_gt_one_hot'):
      training_manager.build_seg_network_gt_one_hot()
      print("Bulding: seg_network_gt_one_hot")

    else:
      if hasattr(config, 'seg_network_gt_one_hot_join'):
        training_manager.build_seg_network_gt_one_hot_join()
        print("Bulding: seg_network_gt_one_hot_join")

      else:
        if hasattr(config, 'rgb_seg_network_enet'):
          training_manager.build_rgb_seg_network_enet()
          print("Bulding: rgb_seg_network_enet")

        else:
          if hasattr(config, 'rgb_seg_network_enet_one_hot'):
            training_manager.build_rgb_seg_network_enet_one_hot()
            print("Bulding: rgb_seg_network_enet_one_hot")

          else:
            if hasattr(config, 'seg_network_enet_one_hot'):
              training_manager.build_seg_network_enet_one_hot()
              print("Bulding: seg_network_enet_one_hot")

            else:
              if hasattr(config, 'seg_network_erfnet_one_hot'):
                training_manager.build_seg_network_erfnet_one_hot() 
                print("Bulding: seg_network_erfnet_one_hot")

              else:
                training_manager.build_network()
                print("Bulding: standard_network")


  """ Initializing Session as variables that control the session """
 
  return training_manager


class DeepRCMachine(Driver):



  # Initializes
  def __init__(self,gpu_number,experiment_name,driver_conf,memory_fraction=0.95):
    
    Driver.__init__(self)


    self._augment_left_right = driver_conf.augment_left_right
 
    self._augmentation_camera_angles = driver_conf.camera_angle # The angle between the cameras used for augmentation and the central camera

    self.stream_id = 0 
    #Rate at which data stream from controller is received
    self.stream_rate = 30
    #Enable/Disable data stream
    self.stream_on = 1

    self.frame_counter = 0

    self._resolution = driver_conf.input_resolution
    self._image_cut = driver_conf.image_cut

    conf_module  = __import__(experiment_name)
    self._config = conf_module.configInput()
    
    config_gpu = tf.ConfigProto()

    config_gpu.gpu_options.per_process_gpu_memory_fraction=memory_fraction
    config_gpu.gpu_options.visible_device_list=gpu_number
    self._sess = tf.Session(config=config_gpu)
   
    #Input Vars
    # This are the channels from the rc controler that represent each of these variables
    self.ch_steer = 0
    self.ch_gas = 2
    self.ch_mode = 7
    self.ch_record = 4
    self.ch_direction = 5
    self.ch_stop = 6

    logging.debug("Starting the CallBack")
    self.callbacks = DeepRCCallbacks()

    #self._mean_image = np.load('data_stats/'+ self._config.dataset_name + '_meanimage.npy')
    self._train_manager =  load_system(conf_module.configTrain())


    self._sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    self._control_function =getattr(machine_output_functions, self._train_manager._config.control_mode )


    cpkt = restore_session(self._sess,saver,self._config.models_path)

    #Output Vars
    logging.debug("Starting ROS")
    #print self.callbacks.rcIn
    rospy.init_node('commFC')
    
    logging.debug("Subscribing to topics")
    #Publisher to send commands
    self.cmd = rospy.Publisher('mavros/rc/override', OverrideRCIn, queue_size=5)

    #Subscribers to receive data    
    rospy.Subscriber("/mavros/rc/in", RCIn, self.callbacks.RCIn)
    rospy.Subscriber("/mavros/rc/out", RCOut, self.callbacks.RCOut)
    rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.callbacks.GPGlobal)
    #rospy.Subscriber("/mavros/global_position/raw/fix", NavSatFix, self.callbackGPRawFix)
    #rospy.Subscriber("/mavros/global_position/local", PoseWithCovarianceStamped, self.callbackGPLocal)
    rospy.Subscriber("/mavros/global_position/gp_vel", TwistStamped, self.callbacks.GPGpVel)
    rospy.Subscriber("/mavros/global_position/raw/gps_vel", TwistStamped, self.callbacks.GPRawGpsVel)
    rospy.Subscriber("/mavros/local_position/velocity", TwistStamped, self.callbacks.LPVelocity)
    #rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.callbackGPRelAlt)
    rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, self.callbacks.GPCompassHdg)
    rospy.Subscriber("/mavros/imu/data", Imu, self.callbacks.IMUData)
    #rospy.Subscriber("/mavros/imu/data_raw", Imu, callbackIMUDataRaw)
    rospy.Subscriber("/mavros/imu/mag", MagneticField, self.callbacks.IMUMag)
    rospy.Subscriber("/mavros/imu/temperature", Temperature, self.callbacks.IMUTemperature)
    rospy.Subscriber("/mavros/imu/atm_pressure", FluidPressure, self.callbacks.IMUAtmPressure)
    #rospy.Subscriber("/mavros/local_position/pose", PoseStamped, callbackLPPose)
    rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self.callbacks.callbackVFRHud)


  def start(self):

    logging.debug("Starting Cameras")
    self.camera_center = cv2.VideoCapture(0) # center
    #self.camera_left = cv2.VideoCapture(0) # left 
    #self.camera_right = cv2.VideoCapture(2) # Right



    #self.camera_left.set(3,self._resolution[0])
    #self.camera_left.set(4,self._resolution[1])
    #self.camera_left.set(6, cv2.VideoWriter_fourcc(*'YUYV')) #or: MJPG
    self.camera_center.set(3,self._resolution[0])
    self.camera_center.set(4,self._resolution[1])
    self.camera_center.set(6, cv2.VideoWriter_fourcc(*'YUYV')) #or: MJPG
    #self.camera_right.set(3,self._resolution[0])
    #self.camera_right.set(4,self._resolution[1])
    #self.camera_right.set(6, cv2.VideoWriter_fourcc(*'YUYV')) #or: MJPG

    # Start the ros interface with the car
    rospy.wait_for_service('/mavros/set_mode')
    set_stream_rate = rospy.ServiceProxy('/mavros/set_stream_rate', StreamRate)
    change_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)

    set_stream_rate(self.stream_id, self.stream_rate, self.stream_on)
    print ("Data stream rate set to: " + str(self.stream_rate))

    change_mode_status = change_mode(custom_mode="manual")
    print ("Mode change " + str(change_mode_status))


    self._rate = rospy.Rate(self.stream_rate) #Run ROS at same frequency as data stream

  def get_recording(self):
    if self.callbacks.rcIn[self.ch_record] < 1500:
      return False
    else:
      return True

  def get_reset(self):
    if self.callbacks.rcIn[self.ch_stop] < 1500:
      return False
    else:
      return True

  def get_direction(self):
    if self.callbacks.rcIn[self.ch_direction] < 1250:
      return 3 #left
    if self.callbacks.rcIn[self.ch_direction] > 1750:
      return 4 #right
    else:
      return 5 #straigt
 
  def compute_action(self,sensor,speed):

    """ Get Steering """
    # receives from 1000 to 2000 
    direction = self.get_direction()

    # Just taking the center image to send to the network


    sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:]

    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    #print future_image

    #image_input = image_input - self._mean_image
    #print "2"
    #image_input = np.multiply(image_input, 1.0 / 127.0)
    image_input = np.multiply(image_input, 1.0 / 255.0)

    if (self._train_manager._config.control_mode == 'single_branch_wp'):

      steer,acc,brake,wp1angle,wp2angle = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)

      steer_pred = steer

      steer_gain = 0.8
      steer = steer_gain*wp1angle
      if steer > 0:
        steer = min(steer,1)
      else:
        steer = max(steer,-1)

      print('Predicted Steering: ',steer_pred, ' Waypoint Steering: ', steer)

    else:

      steer,acc,brake = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)
    
    control = Control()
    control.steer = steer
    control.gas =acc
    control.brake =0

    control.hand_brake = 0
    control.reverse = 0

    if self._augment_left_right: # If augment data, we generate copies of steering for left and right
      control_left = copy.deepcopy(control)

      control_left.steer = self._adjust_steering(control_left.steer,self._augmentation_camera_angles,speed) # The angles are inverse.
      control_right = copy.deepcopy(control)

      control_right.steer = self._adjust_steering(control_right.steer,-self._augmentation_camera_angles,speed)

      return [control,control,control]

    else:
      return [control]


  def get_sensor_data(self):

      ### GET THE SENSOR PART
    
    self._rate.sleep()
    measurements = Measurements()
    measurements.direction = self.get_direction()
    measurements.gps_lat = float(self.callbacks.gpsLat)
    measurements.gps_long = float(self.callbacks.gpsLong)
    measurements.gps_alt = float(self.callbacks.gpsAlt)
    measurements.fused_linear_vel_x = float(self.callbacks.fusedLinearVel.x)
    measurements.fused_linear_vel_y = float(self.callbacks.fusedLinearVel.y)
    measurements.fused_linear_vel_z = float(self.callbacks.fusedLinearVel.z)
    measurements.fused_angular_vel_x = float(self.callbacks.fusedAngularVel.x)
    measurements.fused_angular_vel_y = float(self.callbacks.fusedAngularVel.y)
    measurements.fused_angular_vel_z = float(self.callbacks.fusedAngularVel.z)
    measurements.gps_linear_vel_x = float(self.callbacks.gpsLinearVel.x)
    measurements.gps_linear_vel_y = float(self.callbacks.gpsLinearVel.y)
    measurements.gps_linear_vel_z = float(self.callbacks.gpsLinearVel.z)
    measurements.gps_angular_vel_x =  float(self.callbacks.gpsAngularVel.x)
    measurements.gps_angular_vel_y =  float(self.callbacks.gpsAngularVel.y)
    measurements.gps_angular_vel_z =  float(self.callbacks.gpsAngularVel.z)
    measurements.local_linear_vel_x = float(self.callbacks.localLinearVel.x)
    measurements.local_linear_vel_y = float(self.callbacks.localLinearVel.y)
    measurements.local_linear_vel_z = float(self.callbacks.localLinearVel.z)
    measurements.local_angular_vel_x = float(self.callbacks.localAngularVel.x)
    measurements.local_angular_vel_y = float(self.callbacks.localAngularVel.y)
    measurements.local_angular_vel_z = float(self.callbacks.localAngularVel.z)
    measurements.mag_heading = float(self.callbacks.magHeading)
    measurements.imu_mag_field_x = float(self.callbacks.imuMagField.x)
    measurements.imu_mag_field_y = float(self.callbacks.imuMagField.y)
    measurements.imu_mag_field_z = float(self.callbacks.imuMagField.z)
    measurements.imu_angular_vel_x =float(self.callbacks.imuAngularVel.x)
    measurements.imu_angular_vel_y =float(self.callbacks.imuAngularVel.y)   
    measurements.imu_angular_vel_z =float(self.callbacks.imuAngularVel.z)
    measurements.imu_linear_acc_x = float(self.callbacks.imuLinearAcc.x)
    measurements.imu_linear_acc_y = float(self.callbacks.imuLinearAcc.y)
    measurements.imu_linear_acc_z = float(self.callbacks.imuLinearAcc.z)
    # Quartenions
    measurements.imu_orientation_a = float(self.callbacks.imuOrientation.x)
    measurements.imu_orientation_b = float(self.callbacks.imuOrientation.y)
    measurements.imu_orientation_c = float(self.callbacks.imuOrientation.z)
    measurements.imu_orientation_d = float(self.callbacks.imuOrientation.w)  

    ### GET THE CAMERA PART

    #print measurements


    #ret1, frame_left = self.camera_left.read()
    ret2, frame_center = self.camera_center.read()
    #ret3, frame_right = self.camera_right.read() 
    measurements.speed = math.sqrt(measurements.gps_linear_vel_x*measurements.gps_linear_vel_x \
     + measurements.gps_linear_vel_y*measurements.gps_linear_vel_y)

    #frame_left = frame_left[:, :, ::-1]
    #frame_right = frame_right[:, :, ::-1]    

    frame_center = frame_center[:, :, ::-1]

    return measurements,[frame_center,frame_center,frame_center]

  
  def act(self,action):
    rc_input = OverrideRCIn()
    if self.get_reset():
      rc_input.channels[self.ch_gas] = 0 #Set reset signal
      rc_input.channels[self.ch_steer] = 0 #Set reset signal
    else:

      gas = max(min(action.gas,0.25),0)
      if action.steer < 0.0:
	steer = action.steer/1.2 -0.2
      else:
	steer = action.steer/0.8 -0.2


      #steer = (action.steer - 0.3)

      rc_input.channels[self.ch_gas] = 500*gas+1500 #Set PWM signal
      rc_input.channels[self.ch_steer] = 500*steer+1500 #Set PWM signal
   
    self.cmd.publish(rc_input)
    
    #rospy.loginfo("Sending: %s", rc_input)
    return

  def compute_perception_activations(self,sensor,speed):

    sensor = sensor[self._image_cut[0]:self._image_cut[1],:,:]

    sensor = scipy.misc.imresize(sensor,[self._config.network_input_size[0],self._config.network_input_size[1]])

    image_input = sensor.astype(np.float32)

    image_input = np.multiply(image_input, 1.0 / 255.0)

    #vbp_image =  machine_output_functions.vbp(image_input,speed,self._config,self._sess,self._train_manager)
    vbp_image =  machine_output_functions.seg_viz(image_input,speed,self._config,self._sess,self._train_manager)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #vbp_image = min_max_scaler.fit_transform(np.squeeze(vbp_image))

    #print vbp_image.shape
    return 0.5*grayscale_colormap(np.squeeze(vbp_image),'jet') + 0.5*image_input
