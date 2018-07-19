import RPi.GPIO as GPIO
from time import sleep
import time, math

GPIO.setwarnings(False)

TRIG = 19
ECHO = 21

def degree():
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(TRIG,GPIO.OUT)
    GPIO.setup(ECHO,GPIO.IN)


    GPIO.output(TRIG, False)

    time.sleep(0.00002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration*17150

    distance = round(distance, 2)

    angle = distance * 6

    print"Distance: ", distance, "cm"
    print"Angle: ", angle, "degree"

    return angle
    

   
'''if __name__ == '__main__':
    
    while True:
        degree()
        sleep(0.01)'''

