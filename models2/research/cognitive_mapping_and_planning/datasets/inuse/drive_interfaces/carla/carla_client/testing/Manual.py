#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:58:23 2017

@author: german
"""

from Runnable import *

class Manual(Runnable):
    def run_step(self, data,target):
        control = Control()
        control.steer = 0.0
        control.throttle = 0.9
        control.brake = 0.0
        control.hand_brake = False
        control.reverse = False

        return control
