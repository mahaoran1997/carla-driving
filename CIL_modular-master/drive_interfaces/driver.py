class Driver(object):
    # Initializes all necessary shared data for this class
    def __init__(self):
        pass

    # Should Basically start the system. Any connection needed with the driving device or the
    # data interface should be made on this function.
    # No Return
    def start(self):
        pass

    # @returns: a flag, if true, we should record the sensor data and actions.

    def get_recording(self):
        pass

    # This function is were you should get the data and return it
    # @returns a vector [measurements,images] where:
    # -- measurements -> is a filled object of the class defined above in this file,
    #     it should contain data from all collected sensors
    # -- images -> is the vector of collected images

    def get_sensor_data(self):
        pass

    # this is the function used to send the actions to the car
    # @param: an object of the control class.

    def act(self, control):
        pass
