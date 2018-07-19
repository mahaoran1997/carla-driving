#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import socket, struct
import pickle
import gzip
import binascii
import time
import io
from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from PIL import Image


class Control:
	steer = 0
	gas =0
	brake =0
	hand_brake = 0
	reverse = 0




class Reward:

	position = 0
	speed =  0
	yaw_rate = 0
	yaw = 0
	collided = 0
	lane = 0
	reseted = 0
	goal = 0
	direction = 0
	dist_to_goal = 0
	noise = 0




def frame2numpy(frame, frameSize):
	return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))


def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))

class Scenario:
	def __init__(self, location=None, time=None, weather=None, vehicle=None, drivingMode=None):
		self.location = location #[x,y]
		self.time = time #[hour, minute]
		self.weather = "CLEAR" #string
		self.vehicle = vehicle #string
		self.drivingMode = drivingMode #[drivingMode, setSpeed]


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper

class Start:
	def __init__(self, scenario=None, dataset=None):
		self.scenario = scenario
		self.dataset = dataset

	def to_json(self):
		_scenario = None
		_dataset = None

		if (self.scenario != None):
			_scenario = self.scenario.__dict__

		if (self.dataset != None):
			_dataset = self.dataset.__dict__			

		return json.dumps({'start':{'scenario': _scenario, 'dataset': _dataset}})


class StreamClient:
	def __init__(self, ip='localhost', port=8000, datasetPath=None, compressionLevel=0):
		print('Trying to connect to Streaming')
		self._data_buffer = Queue(1)
		self._current_message =None
		self._sync_time = time.time()


		try:
			self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			print "connect: ",ip," :",port
			self.s.connect((ip, int(port)))
		except:
			print('ERROR: Failed to connect to GTA Streaming')
		else:
			print('Successfully connected to GTA Streaming')
		self.scenario = Scenario(drivingMode=-1) #manual driving
		
		#Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
		self.sendMessage(Start(scenario=self.scenario))
		print 'SEND START'

	def recvMessage(self):
		frame = self._recvall()
		print ("RECEIVED MESSAGE") 
		if not frame: 
			print('ERROR: Failed to receive frame')		
			return None
		data = self._recvall()
		if not data: 
			print('ERROR: Failed to receive message')		
			return None

		dct = json.loads(data.decode('utf-8'))
		#print jsonstr
		dct['frame'] = frame

		return dct

	def sendMessage(self, message):
		jsonstr = message.to_json().encode('utf-8')
		print jsonstr
		try:
			#self.s.sendall(len(jsonstr).to_bytes(4, byteorder='little'))
			self.s.sendall(int2bytes(len(jsonstr)))
			self.s.sendall(jsonstr)
			print jsonstr
		except Exception as e:
			print('ERROR: Failed to send message. Reason:', e)
			return False
		return True


	def recv_bin(self):
		#Receive first size of message in bytes
		data = b""
		while len(data) < 4:
			packet = self.s.recv(4 - len(data))
			if not packet: return None
			data += packet
		size = struct.unpack('I', data)[0]
		return size


	def _recvall(self):
		#Receive first size of message in bytes
		data = b""
		while len(data) < 4:
			packet = self.s.recv(4 - len(data))
			if not packet: return None
			data += packet
		size = struct.unpack('I', data)[0]

		#We now proceed to receive the full message
		data = b""
		while len(data) < size:
			packet = self.s.recv(size - len(data))
			if not packet: return None
			data += packet
		return data

	def close(self):
		self.s.close()

	def get_message(self):
		try:
			message = self._data_buffer.get(timeout=None)
			self._current_message =message
		except Empty:
			pass
		return self._current_message




	@threaded
	def run(self):


		while True:

			try:
				print "Read message"
				self._data_buffer.put(self.recvMessage(),timeout=0)
			except Full:
				pass
			
