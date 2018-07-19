
import numpy as np
import cv2
import mavros_msgs as msgs

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

import copy
from driver import *
import logging

from deeprc_callbacks import *

print(cv2.__version__)

class DeepRCHuman(Driver):



  # Initializes
  def __init__(self,drive_conf):
    
    Driver.__init__(self)

    self._augment_left_right = drive_conf.augment_left_right
 
    self._augmentation_camera_angles = drive_conf.camera_angle # The angle between the cameras used for augmentation and the central camera

    STREAM_ALL=0
    STREAM_RAW_SENSORS=1
    STREAM_EXTENDED_STATUS=2
    STREAM_RC_CHANNELS=3
    STREAM_RAW_CONTROLLER=4
    STREAM_POSITION=6
    STREAM_EXTRA1=10
    STREAM_EXTRA2=11
    STREAM_EXTRA3=12

    self.stream_id = STREAM_ALL 
    #Rate at which data stream from controller is received
    self.stream_rate = 20
    #Enable/Disable data stream
    self.stream_on = 1

    self.frame_counter = 0

    self._resolution = drive_conf.resolution


    #Input Vars
    # This are the channels from the rc controler that represent each of these variables
    self.ch_steer = 0
    self.ch_steer_remote = 1
    self.ch_gas = 2
    self.ch_gas_remote = 3
    self.ch_mode = 7
    self.ch_record = 4
    self.ch_direction = 5
    self.ch_stop = 6

    logging.debug("Starting the CallBack")
    self.callbacks = DeepRCCallbacks()

    #Output Vars
    logging.debug("Starting ROS")
    print self.callbacks.rcIn
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
    self.camera_center = cv2.VideoCapture(1) # center
    self.camera_left = cv2.VideoCapture(0) # left 
    self.camera_right = cv2.VideoCapture(2) # Right

    self.camera_left.set(3,self._resolution[0])
    self.camera_left.set(4,self._resolution[1])
    self.camera_left.set(6, cv2.VideoWriter_fourcc(*'YUYV')) #or: MJPG
    self.camera_center.set(3,self._resolution[0])
    self.camera_center.set(4,self._resolution[1])
    self.camera_center.set(6, cv2.VideoWriter_fourcc(*'YUYV')) #or: MJPG
    self.camera_right.set(3,self._resolution[0])
    self.camera_right.set(4,self._resolution[1])
    self.camera_right.set(6, cv2.VideoWriter_fourcc(*'YUYV')) #or: MJPG

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

    if self.get_reset():
      #print("Manual Control")
      steering_axis = self.callbacks.rcIn[self.ch_steer]
      gas_axis =  self.callbacks.rcIn[self.ch_gas]

    else: #Control through TX2
      #print("TX2 Control")
      steering_axis = self.callbacks.rcIn[self.ch_steer_remote]
      gas_axis =  self.callbacks.rcIn[self.ch_gas_remote]

    control = Control()
  
    control.steer = (float(steering_axis) -1500.0)/500.0 # Scale properly
    control.gas = (float(gas_axis) -1500.0)/500.0   
    control.brake = 0  

    #print 'Steering Axis'
    #print steering_axis, control.steer
      
    #print 'Gas Axis'
    #print gas_axis,control.gas
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

    measurements.imu_orientation_a = float(self.callbacks.imuOrientation.x)
    measurements.imu_orientation_b = float(self.callbacks.imuOrientation.y)
    measurements.imu_orientation_c = float(self.callbacks.imuOrientation.z)
    measurements.imu_orientation_d = float(self.callbacks.imuOrientation.w) 

    measurements.vrf_hud_airspeed = float(self.callbacks.vrf_hudAirspeed)
    measurements.vrf_hud_groundspeed = float(self.callbacks.vrf_hudGroundspeed)
    measurements.vrf_hud_heading = float(self.callbacks.vrf_hudHeading)
    measurements.vrf_hud_throttle = float(self.callbacks.vrf_hudThrottle)
    measurements.vrf_hud_altitude = float(self.callbacks.vrf_hudAltitude)   
    measurements.vrf_hud_climb = float(self.callbacks.vrf_hudClimb)  

    ### GET THE CAMERA PART

    #print measurements
    #print measurements.local_linear_vel_x,measurements.local_linear_vel_y,measurements.local_linear_vel_z

    ret1, frame_left = self.camera_left.read()
    ret2, frame_center = self.camera_center.read()
    ret3, frame_right = self.camera_right.read() 


    frame_left = frame_left[:, :, ::-1]

    frame_center = frame_center[:, :, ::-1]

    frame_right = frame_right[:, :, ::-1]    


    return measurements,[frame_left,frame_center,frame_right]

  
  def act(self,action):
    rc_input = OverrideRCIn()

    if self.get_reset():
      rc_input.channels[self.ch_gas] = 0 #Set reset signal
      rc_input.channels[self.ch_steer] = 0 #Set reset signal

    else: #Control through TX2
      rc_input.channels[self.ch_gas] = 500*action.gas+1500 #Set PWM signal
      rc_input.channels[self.ch_steer] = 500*action.steer+1500 #Set PWM signal
      #print("Sending: ", rc_input)
    self.cmd.publish(rc_input)
    return
