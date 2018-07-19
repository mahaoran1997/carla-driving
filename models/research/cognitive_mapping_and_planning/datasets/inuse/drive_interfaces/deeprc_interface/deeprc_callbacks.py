class Vector3:
    x=0
    y=0
    z=0

class Vector4:
    x=0
    y=0
    z=0
    w=0

class Measurements:
	
    direction = 0
    gps_lat = 0
    gps_long = 0
    gps_alt = 0
    fused_linear_vel_x = 0
    fused_linear_vel_y = 0
    fused_linear_vel_z = 0
    fused_angular_vel_x = 0
    fused_angular_vel_y = 0
    fused_angular_vel_z = 0
    gps_linear_vel_x = 0
    gps_linear_vel_y = 0
    gps_linear_vel_z = 0
    gps_angular_vel_x = 0
    gps_angular_vel_y = 0
    gps_angular_vel_z = 0
    local_linear_vel_x = 0
    local_linear_vel_y = 0
    local_linear_vel_z = 0
    local_angular_vel_x = 0
    local_angular_vel_y = 0
    local_angular_vel_z = 0
    mag_heading = 0
    imu_mag_field_x = 0
    imu_mag_field_y = 0
    imu_mag_field_z = 0
    imu_angular_vel_x = 0
    imu_angular_vel_y = 0 
    imu_angular_vel_z = 0
    imu_linear_acc_x = 0
    imu_linear_acc_y = 0
    imu_linear_acc_z = 0
    # Quartenions
    imu_orientation_a = 0
    imu_orientation_b = 0
    imu_orientation_c = 0
    imu_orientation_d = 0
    vrf_hud_airspeed = 0
    vrf_hud_groundspeed = 0
    vrf_hud_heading = 0
    vrf_hud_throttle = 0 
    vrf_hud_altitude = 0   
    vrf_hud_climb = 0  
    speed = 0 
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Control:
    steer = 0
    gas = 0
    brake =0
    hand_brake = 0
    reverse = 0



import rospy

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

#/mavros/rc/in - Publish RC inputs (in raw milliseconds) 
from mavros_msgs.msg import RCIn

#/mavros/rc/out - Publish FCU servo outputs 
from mavros_msgs.msg import RCOut
  
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

class DeepRCCallbacks(object):


  def __init__(self):


    self.bDebug = 0
    self.rcIn = [0,0,0,0,0,0,0,0]
    self.rcOut = [0,0,0,0,0,0,0,0]
    self.gpsLat = 0
    self.gpsLong = 0
    self.gpsAlt = 0
    self.fusedLinearVel = Vector3()
    self.fusedAngularVel = Vector3()
    self.gpsLinearVel = Vector3()
    self.gpsAngularVel = Vector3()
    self.localLinearVel = Vector3()
    self.localAngularVel = Vector3()
    self.magHeading = 0
    self.imuOrientation =Vector4()
    self.imuAngularVel = Vector3()
    self.imuLinearAcc = Vector3()
    self.imuMagField = Vector3()
    self.vrf_hudAirspeed = 0
    self.vrf_hudGroundspeed = 0
    self.vrf_hudHeading = 0
    self.vrf_hudThrottle = 0
    self.vrf_hudAltitude = 0
    self.vrf_hudClimb = 0

  def RCIn(self,RCIn):
    if self.bDebug:
        rospy.loginfo("RSSI: %s, RC In: %s \n", RCIn.rssi, RCIn.channels)
    if len(RCIn.channels) ==0:
	print "Empty Tuple Received from the RC controler, you maybe need to turn it on"
    else:    
	self.rcIn = RCIn.channels
    return 

  def RCOut(self, RCOut):
    if self.bDebug:
        rospy.loginfo("RC Out: %s \n", RCOut.channels)
    self.rcOut = RCOut.channels
    return  

  def GPGlobal(self, NavSatFix):
    if self.bDebug:
        rospy.loginfo("Latitude: %s, Longitude: %s, Altitude: %s \n", NavSatFix.latitude, NavSatFix.longitude, NavSatFix.altitude)
    self.gpsLat = NavSatFix.latitude
    self.gpsLong = NavSatFix.longitude
    self.gpsAlt = NavSatFix.altitude
    return 

  def GPGpVel(self, TwistStamped):
    if self.bDebug:
        rospy.loginfo("Linear Veloclity: %s, Angular Velocity: %s \n", TwistStamped.twist.linear, TwistStamped.twist.angular)
    self.fusedLinearVel = TwistStamped.twist.linear
    self.fusedAngularVel = TwistStamped.twist.angular
    return 

  def GPRawGpsVel(self,TwistStamped):
    if self.bDebug:
        rospy.loginfo("Linear Veloclity: %s, Angular Velocity: %s \n", TwistStamped.twist.linear, TwistStamped.twist.angular)
    self.gpsLinearVel = TwistStamped.twist.linear
    self.gpsAngularVel = TwistStamped.twist.angular
    return 

  def LPVelocity(self, TwistStamped):
    if self.bDebug:
        rospy.loginfo("Linear Veloclity: %s, Angular Velocity: %s \n", TwistStamped.twist.linear, TwistStamped.twist.angular)
    self.localLinearVel = TwistStamped.twist.linear
    self.localAngularVel = TwistStamped.twist.angular
    return 

  def GPCompassHdg(self, Float64):
    if self.bDebug:
        rospy.loginfo("Heading: %s \n", Float64.data)
    self.magHeading = Float64.data
    return 

  def IMUData(self, Imu):
    if self.bDebug:
        rospy.loginfo("Orientation: %s, Angular Velocity: %s, Linear Acceleration: %s \n", Imu.orientation, Imu.angular_velocity, Imu.linear_acceleration)
    self.imuOrientation = Imu.orientation
    self.imuAngularVel = Imu.angular_velocity
    self.imuLinearAcc = Imu.linear_acceleration
    return 

  def IMUMag(self, MagneticField):
    if self.bDebug:
        rospy.loginfo("Magnetic Field: %s \n", MagneticField.magnetic_field)
    self.imuMagField = MagneticField.magnetic_field
    return 

  def IMUTemperature(self, Temperature):
    if self.bDebug:
        rospy.loginfo("Temperature: %s \n", Temperature.temperature)
    return 

  def IMUAtmPressure(self, FluidPressure):
    if self.bDebug:
        rospy.loginfo("Pressure [Pa]: %s \n", FluidPressure.fluid_pressure)
    return

  def callbackVFRHud (self, VFR_HUD):
    if self.bDebug:
        rospy.loginfo("Airspeed: %s, Groundspeed: %s, Heading: %s \n", VFR_HUD.airspeed, VFR_HUD.groundspeed, VFR_HUD.heading)
    self.vrf_hudAirspeed = VFR_HUD.airspeed
    self.vrf_hudGroundspeed = VFR_HUD.groundspeed
    self.vrf_hudHeading = VFR_HUD.heading
    self.vrf_hudThrottle = VFR_HUD.throttle
    self.vrf_hudAltitude = VFR_HUD.altitude
    self.vrf_hudClimb = VFR_HUD.climb
    return
