#!/usr/bin/env python
#message: mavros/OverrideRCIn
#topic: mavros/rc/override

import numpy as np
import cv2
import rospy
import mavros_msgs as msgs

#/mavros/rc/override - Override RC inputs
from mavros_msgs.msg import OverrideRCIn

#/mavros/set_stream_rate - Send stream rate request to FCU
from mavros_msgs.srv import StreamRate 

#/mavros/set_mode - Set FCU operation mode (http://wiki.ros.org/mavros/CustomModes) 
from mavros_msgs.srv import SetMode

#/mavros/rc/in - Publish RC inputs (in raw milliseconds) 
from mavros_msgs.msg import RCIn

#/mavros/rc/out - Publish FCU servo outputs 
from mavros_msgs.msg import RCOut  

#/mavros/global_position/global - GPS Fix
#/mavros/global_position/raw/fix - GPS position fix reported by the device
from sensor_msgs.msg import NavSatFix

#/mavros/global_position/local - UTM coords
from geometry_msgs.msg import PoseWithCovarianceStamped

#/mavros/global_position/gp_vel - Velocity fused by FCU
#/mavros/global_position/raw/gps_vel - Velocity output from the GPS device
#/mavros/local_position/velocity - Velocity data from FCU
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

#/mavros/global_position/rel_alt - Relative altitude
#/mavros/global_position/compass_hdg - Compass heading in degrees
from std_msgs.msg import Float64

#/mavros/imu/data - Imu data, orientation computed by FCU
#/mavros/imu/data_raw - Raw IMU data without orientation
from sensor_msgs.msg import Imu  
from geometry_msgs.msg import Quaternion

#/mavros/imu/mag - FCU compass data 
from sensor_msgs.msg import MagneticField
  
#/mavros/imu/temperature - Temperature reported by FCU (usually from barometer) 
from sensor_msgs.msg import Temperature
  
#/mavros/imu/atm_pressure - Air pressure
from sensor_msgs.msg import FluidPressure

#/mavros/local_position/pose - Local position from FCU
from geometry_msgs.msg import PoseStamped

#/mavros/vfr_hud - Data for HUD
from mavros_msgs.msg import VFR_HUD

print(cv2.__version__)

#Open cameras
#cap1 = cv2.VideoCapture(1)
#cap2 = cv2.VideoCapture(2)
#cap3 = cv2.VideoCapture(3)

class rosInterface:
  def __init__(self):
    #0 - Enable all data streams
    #1 - Enable IMU_RAW, GPS_RAW, GPS_STATUS packets
    #2 - Enable GPS_STATUS, CONTROL_STATUS, AUX_STATUS
    #3 - Enable RC_CHANNELS_SCALED, RC_CHANNELS_RAW, SERVO_OUTPUT_RAW
    #6 - Enable LOCAL_POSITION, GLOBAL_POSITION/GLOBAL_POSITION_INT messages
    self.stream_id = 0 
    #Rate at which data stream from controller is received
    self.stream_rate = 1 
    #Enable/Disable data stream
    self.stream_on = 1

    self.frame_counter = 0

    #Input Vars
    self.ch_steer = 0
    self.ch_gas = 2
    self.ch_mode = 7
    self.ch_record = 4
    self.ch_direction = 5
    self.ch_stop = 6



    self.bDebug = 0

    #Output Vars
    self.rcIn = [0,0,0,0,0,0,0,0] # 


    self.rcOut = [0,0,0,0,0,0,0,0] 
    self.gpsLat = 0
    self.gpsLong = 0
    self.gpsAlt = 0
    self.fusedLinearVel = 0
    self.fusedAngularVel = 0
    self.gpsLinearVel = 0
    self.gpsAngularVel = 0
    self.localLinearVel = 0
    self.localAngularVel = 0
    self.magHeading = 0
    self.imuOrientation = [0,0,0,0]
    self.imuAngularVel = [0,0,0]
    self.imuLinearAcc = [0,0,0]
    self.imuMagField = [0,0,0]

    rospy.init_node('commFC')

    #Publisher to send commands
    self.cmd = rospy.Publisher('mavros/rc/override', OverrideRCIn, queue_size=5)

    #Subscribers to receive data    
    rospy.Subscriber("/mavros/rc/in", RCIn, self.callbackRCIn)
    rospy.Subscriber("/mavros/rc/out", RCOut, self.callbackRCOut)
    rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.callbackGPGlobal)
    #rospy.Subscriber("/mavros/global_position/raw/fix", NavSatFix, self.callbackGPRawFix)
    #rospy.Subscriber("/mavros/global_position/local", PoseWithCovarianceStamped, self.callbackGPLocal)
    rospy.Subscriber("/mavros/global_position/gp_vel", TwistStamped, self.callbackGPGpVel)
    rospy.Subscriber("/mavros/global_position/raw/gps_vel", TwistStamped, self.callbackGPRawGpsVel)
    rospy.Subscriber("/mavros/local_position/velocity", TwistStamped, self.callbackLPVelocity)
    #rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.callbackGPRelAlt)
    rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, self.callbackGPCompassHdg)
    rospy.Subscriber("/mavros/imu/data", Imu, self.callbackIMUData)
    #rospy.Subscriber("/mavros/imu/data_raw", Imu, callbackIMUDataRaw)
    rospy.Subscriber("/mavros/imu/mag", MagneticField, self.callbackIMUMag)
    rospy.Subscriber("/mavros/imu/temperature", Temperature, self.callbackIMUTemperature)
    rospy.Subscriber("/mavros/imu/atm_pressure", FluidPressure, self.callbackIMUAtmPressure)
    #rospy.Subscriber("/mavros/local_position/pose", PoseStamped, callbackLPPose)
    #rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, callbackVFRHud)

  def callbackRCIn(self,RCIn):
    if self.bDebug:
        rospy.loginfo("RSSI: %s, RC In: %s \n", RCIn.rssi, RCIn.channels)
    self.rcIn = RCIn.channels
    return 

  def callbackRCOut(self, RCOut):
    if self.bDebug:
        rospy.loginfo("RC Out: %s \n", RCOut.channels)
    self.rcOut = RCOut.channels
    return  

  def callbackGPGlobal(self, NavSatFix):
    if self.bDebug:
        rospy.loginfo("Latitude: %s, Longitude: %s, Altitude: %s \n", NavSatFix.latitude, NavSatFix.longitude, NavSatFix.altitude)
    self.gpsLat = NavSatFix.latitude
    self.gpsLong = NavSatFix.longitude
    self.gpsAlt = NavSatFix.altitude
    return 

  def callbackGPGpVel(self, TwistStamped):
    if self.bDebug:
        rospy.loginfo("Linear Veloclity: %s, Angular Velocity: %s \n", TwistStamped.twist.linear, TwistStamped.twist.angular)
    self.fusedLinearVel = TwistStamped.twist.linear
    self.fusedAngularVel = TwistStamped.twist.angular
    return 

  def callbackGPRawGpsVel(self,TwistStamped):
    if self.bDebug:
        rospy.loginfo("Linear Veloclity: %s, Angular Velocity: %s \n", TwistStamped.twist.linear, TwistStamped.twist.angular)
    self.gpsLinearVel = TwistStamped.twist.linear
    self.gpsAngularVel = TwistStamped.twist.angular
    return 

  def callbackLPVelocity(self, TwistStamped):
    if self.bDebug:
        rospy.loginfo("Linear Veloclity: %s, Angular Velocity: %s \n", TwistStamped.twist.linear, TwistStamped.twist.angular)
    self.localLinearVel = TwistStamped.twist.linear
    self.localAngularVel = TwistStamped.twist.angular
    return 

  def callbackGPCompassHdg(self, Float64):
    if self.bDebug:
        rospy.loginfo("Heading: %s \n", Float64.data)
    self.magHeading = Float64.data
    return 

  def callbackIMUData(self, Imu):
    if self.bDebug:
        rospy.loginfo("Orientation: %s, Angular Velocity: %s, Linear Acceleration: %s \n", Imu.orientation, Imu.angular_velocity, Imu.linear_acceleration)
    self.imuOrientation = Imu.orientation
    self.imuAngularVel = Imu.angular_velocity
    self.imuLinearAcc = Imu.linear_acceleration
    return 

  def callbackIMUMag(self, MagneticField):
    if self.bDebug:
        rospy.loginfo("Magnetic Field: %s \n", MagneticField.magnetic_field)
    self.imuMagField = MagneticField.magnetic_field
    return 

  def callbackIMUTemperature(self, Temperature):
    if self.bDebug:
        rospy.loginfo("Temperature: %s \n", Temperature.temperature)
    return 

  def callbackIMUAtmPressure(self, FluidPressure):
    if self.bDebug:
        rospy.loginfo("Pressure [Pa]: %s \n", FluidPressure.fluid_pressure)
    return 
        
  def sendCmd(self):
    rc_input = OverrideRCIn()
    rc_input.channels[self.ch_gas] = 500*gas+1500 #Set PWM signal
    rc_input.channels[self.ch_steer] = 500*steer+1500 #Set PWM signal
    self.cmd.publish(rc_input)
    if self.bDebug:
        rospy.loginfo("Sending: %s", rc_input)
    return

  def sendReset(self):
    rc_input = OverrideRCIn()
    rc_input.channels[self.ch_gas] = 0 #Set PWM signal
    rc_input.channels[self.ch_steer] = 0 #Set PWM signal
    cmd.publish(rc_input)
    if self.bDebug:
        rospy.loginfo("Sending: %s", rc_input)
    return

  def run(self):
    rospy.wait_for_service('/mavros/set_mode')
    set_stream_rate = rospy.ServiceProxy('/mavros/set_stream_rate', StreamRate)
    change_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)

    set_stream_rate(self.stream_id, self.stream_rate, self.stream_on)
    print ("Data stream rate set to: " + str(self.stream_rate))

    change_mode_status = change_mode(custom_mode="manual")
    print ("Mode change " + str(change_mode_status))

  #  if "True" in str(change_mode_status):
    if (True):
###Receive messages
      rate = rospy.Rate(self.stream_rate) #Run ROS at same frequency as data stream
               
      while not rospy.is_shutdown():
      
        print ("rcIn: %s, rcOut: %s" % (self.rcIn, self.rcOut))
        print ("gpsLat: %s, gpsLong: %s, gpsAlt: %s" % (self.gpsLat, self.gpsLong, self.gpsAlt)) 
        print ("fusedLinearVel: %s, fusedAngularVel: %s" % (self.fusedLinearVel, self.fusedAngularVel))
        print ("gpsLinearVel: %s, gpsAngularVel: %s" % (self.gpsLinearVel, self.gpsAngularVel))
        print ("localLinearVel: %s, localAngularVel: %s" % (self.localLinearVel, self.localAngularVel)) 
        print ("magHeading: %s, imuMagField: %s" % (self.magHeading, self.imuMagField))
        print ("localLinearVel: %s, localAngularVel: %s" % (self.localLinearVel, self.localAngularVel)) 
        print ("imuOrientation: %s, imuAngularVel: %s, imuLinearAcc: %s" % (self.imuOrientation, self.imuAngularVel, self.imuLinearAcc)) 

###Send messages
	#sendCmd(1,1) #Send Control
        #sendReset() #Reset

###Save images
        #frame_counter += 1
        ##Capture frame-by-frame
        #ret1, frame1 = cap1.read()
        #ret2, frame2 = cap2.read()
        #ret3, frame3 = cap3.read()         
        #if self.bDebug:
        #  #Display the resulting frame
        #  cv2.imshow('cam1',frame1)
        #  cv2.imshow('cam2',frame2)
        #  cv2.imshow('cam3',frame3)
        #cv2.imwrite("data\" + str(frame_counter) + "_cam1.jpg", frame1)
        #cv2.imwrite("data\" + str(frame_counter) + "_cam2.jpg", frame2)
        #cv2.imwrite("data\" + str(frame_counter) + "_cam3.jpg", frame3)

        # Keeps python from exiting until nodes are stopped
        rate.sleep()
        #rospy.spin()

if __name__ == '__main__':

  try:    
    ri = rosInterface()
    ri.run()

  except rospy.ROSInterruptException:
  # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
  pass
