import math
import sys
import cv2
from controller import Robot, Keyboard
import numpy as np
import time
def sign(x):
    return (x > 0) - (x < 0)

def clamp(value, low, high):
    return max(low, min(value, high))

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)
front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
keyboard = robot.getKeyboard()
keyboard.enable(timestep)
camera_roll_motor = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")

motors = [
    robot.getDevice("front left propeller"),
    robot.getDevice("front right propeller"),
    robot.getDevice("rear left propeller"),
    robot.getDevice("rear right propeller")
]

for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(1.0)

print("Start the drone...")

while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

print("You can control the drone with your computer keyboard:")
print("- 'up': move forward.")
print("- 'down': move backward.")
print("- 'right': turn right.")
print("- 'left': turn left.")
print("- 'shift + up': increase the target altitude.")
print("- 'shift + down': decrease the target altitude.")
print("- 'shift + right': strafe right.")
print("- 'shift + left': strafe left.")

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 1.0

image_index = 0
capture_images = False
images_to_capture = 2

while robot.step(timestep) != -1:
    time = robot.getTime()

    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    altitude = gps.getValues()[2]
    roll_velocity = gyro.getValues()[0]
    pitch_velocity = gyro.getValues()[1]

    led_state = int(time) % 2
    front_left_led.set(led_state)
    front_right_led.set(not led_state)

    camera_roll_motor.setPosition(0)
    camera_pitch_motor.setPosition(0)

    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0
    
    image = camera.getImage()    
    width, height = camera.getWidth(), camera.getHeight()        
    image_array = np.frombuffer(image, dtype=np.uint8)   
    image_array = image_array.reshape((height, width, 4))

    key = keyboard.getKey()

    while key > 0:
        # Check if the key is one of the specified keys to capture images
        if key in [Keyboard.UP, Keyboard.DOWN, Keyboard.RIGHT, Keyboard.LEFT,
                   (Keyboard.SHIFT + Keyboard.RIGHT), (Keyboard.SHIFT + Keyboard.LEFT),
                   (Keyboard.SHIFT + Keyboard.UP), (Keyboard.SHIFT + Keyboard.DOWN)]:
            capture_images = True
            images_to_capture = 1

        if capture_images:
            if key == Keyboard.UP:
                pitch_disturbance = -2
                type="Forward"
            elif key == Keyboard.DOWN:
                pitch_disturbance = 0.7
                type="Backward"
            elif key == Keyboard.RIGHT:
                yaw_disturbance = -0.7
                type="Right"
            elif key == Keyboard.LEFT:
                yaw_disturbance = 0.7
                type="Left"
            elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
                roll_disturbance = -1.0
            elif key == (Keyboard.SHIFT + Keyboard.LEFT):
                roll_disturbance = 1.0
            elif key == (Keyboard.SHIFT + Keyboard.UP):
                target_altitude += 0.05
                print("target altitude: {} [m]".format(target_altitude))
            elif key == (Keyboard.SHIFT + Keyboard.DOWN):
                target_altitude -= 0.05
                print("target altitude: {} [m]".format(target_altitude))

            # Save Images to build model later
            if image_index % 9 == 0:
                image_name = f"./TrainData/{type}/{key}_{image_index}.png"
                cv2.imwrite(image_name, image_array)
            image_index += 1
            images_to_capture -= 1

            if images_to_capture == 0:
                capture_images = False

        key = keyboard.getKey()

    roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance
    clamped_difference_altitude = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0)

    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

    for i, motor in enumerate(motors):
        if i == 1 or i == 2:
            motor.setVelocity(-front_right_motor_input if i == 1 else -rear_left_motor_input)
        else:
            motor.setVelocity(front_left_motor_input if i == 0 else rear_right_motor_input)


robot.cleanup()
sys.exit(0)
