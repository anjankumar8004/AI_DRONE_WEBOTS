import math
import sys
import cv2
from controller import Robot, Keyboard
import numpy as np 
import tensorflow as tf
import pickle
from keras.models import load_model
from time import sleep as delay
import threading
import random
from queue import Queue  # Added import for Queue

def sign(x):
    return (x > 0) - (x < 0)

def clamp(value, low, high):
    return max(low, min(value, high))

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(64)
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
#Load model which build in prev steps i.e model building.
model = tf.keras.models.load_model("E:/AI Projects/ReInforcement learning/RL_DRONE/Model/drone_cnn_model.h5")
model.load_weights("E:/AI Projects/ReInforcement learning/RL_DRONE/Model/drone_cnn_model_weights.h5")

dict_file = open("./ai_drone.pkl", "rb")
category_dict = pickle.load(dict_file)

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 1.0

image_index = 0

roll_disturbance = 0.0
pitch_disturbance = 0.0
yaw_disturbance = 0.0

image_array = None  # Shared variable for storing camera image data

# Queue for communication between threads
prediction_queue = Queue()

# Function to perform model prediction in a separate thread
def perform_model_prediction(image_array):
    global roll_disturbance, pitch_disturbance, yaw_disturbance,acc
    results = model.predict(image_array)
    label = np.argmax(results, axis=1)[0]
    acc = int(np.max(results, axis=1)[0] * 100)
    #print(acc)
    prediction_queue.put([label,acc])

# Function to continuously capture images from the camera
def model_prediction_thread():
    global image_array

    while True:
        image = camera.getImage()   
        #delay(0.1)
        width, height = camera.getWidth(), camera.getHeight()        
        image_array = np.frombuffer(image, dtype=np.uint8)   
        image_array = image_array.reshape((height, width, 4))

        test_img = cv2.resize(image_array, (50, 50))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img = test_img/255
        test_img = test_img.reshape(1, 50, 50, 1)
        test_img = np.reshape(test_img, (test_img.shape[0], 50, 50, 1))

        # Run model prediction in a separate thread
        perform_model_prediction_thread = threading.Thread(target=perform_model_prediction, args=(test_img,))
        perform_model_prediction_thread.start()
        perform_model_prediction_thread.join()  # Wait for the prediction to finish
     
# Start the model prediction thread
prediction_thread = threading.Thread(target=model_prediction_thread)
prediction_thread.start()


def set_motor_velocity(front_left_motor_input,front_right_motor_input,
    rear_left_motor_input,rear_right_motor_input):
    for i, motor in enumerate(motors):
        if i == 1 or i == 2:
            motor.setVelocity(-front_right_motor_input if i == 1 else -rear_left_motor_input)
            
        else:
            motor.setVelocity(front_left_motor_input if i == 0 else rear_right_motor_input)
        #delay(0.05)
    #delay(0.1)


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

    camera_roll_motor.setPosition(-0.115 * roll_velocity)
    camera_pitch_motor.setPosition(-0.1 * pitch_velocity)
    
    # Get the prediction result from the queue
    if not prediction_queue.empty():
        label,acc = prediction_queue.get()
        if label == 0:    # FORWARD
            pitch_disturbance = -0.4
            print(f"Moving: {category_dict[label]} with {acc}% accuracy.")
        elif label == 1:  # LEFT
            yaw_disturbance=0.3
            roll_disturbance = 0.1
            print(f"Moving: {category_dict[label]} with {acc}% accuracy.")
        elif label == 2:  # RIGHT
            yaw_disturbance=-0.3
            roll_disturbance = -0.1
            print(f"Moving: {category_dict[label]} with {acc}% accuracy.")    
       
    prediction_queue = Queue()
    roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance
    clamped_difference_altitude = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0)
    
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
    
    motor_thread = threading.Thread(target=set_motor_velocity, args=(front_left_motor_input,front_right_motor_input,
    rear_left_motor_input,rear_right_motor_input))

    motor_thread.start()
    motor_thread.join()  # Wait for the motor_thread to finish


robot.cleanup()
sys.exit(0)