import numpy as np
import cv2
#import mavros_msgs as msgs

import pygame

import subprocess

import copy
from driver import *
import logging

#from deeprc_callbacks import *

import socket

camera_port = 0   # Change this to your webcam ID, or file name for your video file
ramp_frames = 15  #Number of frames to throw away while the camera adjusts to light levels
#rval = True

UDP_IP = "10.42.0.144"
UDP_PORT = 5050
sock = socket.socket(socket.AF_INET, # Internet
          socket.SOCK_DGRAM) # UDP

class Control:
    speed = 0
    steer = 0


print(cv2.__version__)

class ElektraHuman(Driver):



  # Initializes
  # Do everything needed to start the system.
  def __init__(self,drive_conf):
    
    Driver.__init__(self)

    self._augment_left_right = drive_conf.augment_left_right
 
    self._augmentation_camera_angles = drive_conf.camera_angle 

    self._recording= False    
    self._rear = False
    self.steering_direction = 0
    self._new_speed = 0




  # Start the communication process and any necessary threads or other process
  def start(self):

    # Starts the communication with raspy
    import subprocess
    print "start"
    subprocess.call("./connect_pi.sh", shell=True)
    print "end"

    # Starts communication with the cameras
    
    # Start any background process... etc.

    # Start the interface with the joystick 
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count >1:
      print "Please Connect Just One Joystick"
      raise 
    print joystick_count
    self.joystick = pygame.joystick.Joystick(0)
    self.joystick.init()


  def get_recording(self):
    # Joystick command to activate and deactivate record
    if( self.joystick.get_button( 2 )):
      self._recording =True
    if( self.joystick.get_button( 1 )):
      self._recording=False
    return self._recording


  def get_reset(self):

    #if( self.joystick.get_button( 8 )):
    # Maybe you need some kind of reset.
    return False


  def get_direction(self):

    return 2.0


  def compute_action(self,sensor,speed):

    self._old_speed = speed
    global start_time

    """ Get Steering """
    if self.joystick.get_button( 6 ):  #left
      self.steering_direction = -1
    elif self.joystick.get_button( 7 ): #right
      self.steering_direction = 1
    else:
      self.steering_direction = 0     #when left or right button is not pressed, bring the steering to centre

    
    if( self.joystick.get_button(3)):  #increase speed
      end_time = datetime.datetime.now()
      time_diff = (end_time - start_time).microseconds / 1000   #in milliseconds
      if time_diff > 300: #to ensure same click isnt counted multiple times
        self._new_speed = self._old_speed + 0.7   #max speed = 7 kmph, changes in 10 steps
        self._new_speed = min(7, self._new_speed) #restrict between 0-7
        start_time = datetime.datetime.now()
    
    if( self.joystick.get_button(0)):  #decrease speed
      end_time = datetime.datetime.now()
      time_diff = (end_time - start_time).microseconds / 1000
      if time_diff > 300:
        self._new_speed = self._old_speed - 0.7
        self._new_speed = max(0, self._new_speed)
        start_time = datetime.datetime.now()


    if( self.joystick.get_button( 10 )):
      self._rear =True
    if( self.joystick.get_button( 9 )):
      self._rear=False


    control = Control()
    control.speed = self._new_speed
    control.steer = self.steering_direction

    return control, self._new_speed



  def get_sensor_data(self):

    # Get the camera image
    camera = cv2.VideoCapture(camera_port)
 
    # Ramp the camera - these frames will be discarded and are only used to allow v4l2 to adjust light levels, if necessary
    for i in xrange(ramp_frames):
      retval, temp = camera.read()
    print("Taking image...")

    # Take the actual image we want to keep
    retval, frame = camera.read()  # Captures a single image from the camera in PIL format

    r=frame.shape[0]
    c=frame.shape[1]
    frame = frame[1:r, 1:c/2] #just take the left camera image

    '''file = "./test_image15.png"
    cv2.imwrite(file, frame)'''

    '''cv2.imshow('image',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
 
    del(camera)  # You'll want to release the camera, otherwise you won't be able to create a new capture object until your script exits



    # get all the measurements the car is making.

    


    return frame


  
  def act(self,control):
  # Send the action to the raspy
    
    #you may have to define self._new_speed and self._old_speed in class definition
    change=(self._new_speed-self._old_speed)/10   ##********10 wont work!!!********
  
    if(change<0):
      print "Control for decrease"
      for k in range(abs(change)):    
        MESSAGE = '<'
        print MESSAGE
        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))  
      MESSAGE = 'w'
      print MESSAGE
      sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))        
  
  
    else: 
      print "Control for increase"    
      for k in range(abs(change)):    
        MESSAGE = '>'
        print MESSAGE
        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))  
      MESSAGE = 'w'
      print MESSAGE
      sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

