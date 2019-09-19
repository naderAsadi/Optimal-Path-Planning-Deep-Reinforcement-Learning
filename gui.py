"""
Created on Mon May 7 20:13:21 2018

@author: nader
"""

import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
#import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the AI
from ai import Dqn

# We use the code below to stop adding red cirles by right clicking
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# we use last_x and last_y to keep track of last point we draw on canvas
last_x = 0
last_y = 0
n_points = 0
length = 0

# Initialization of Brain with 5 sensors and 3 actions and 0.9 discount factor
brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []

# map initialization
first_update = True
def init():
    #sand is an array with size of screen pixel size
    global sand
    global goal_x
    global goal_y
    global first_update
    #longueu,largeu
    #sand = np.zeros((length,width))
    # initializing sand array with 0
    sand=np.zeros((length,width))
    goal_x = 20
    goal_y = width - 20
    first_update = False

last_distance = 0


class Car(Widget):
    
    # initializing the angle of the car
    angle = NumericProperty(0)
    # initializing rotation of the car
    rotation = NumericProperty(0)
    # initializing speed in x-vector
    velocity_x = NumericProperty(0)
    # initializing speed in y-vector
    velocity_y = NumericProperty(0)
    # speed vector
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    # initializing x of forward sensor
    sensor1_x = NumericProperty(0)
    # initializing y of forward sensor
    sensor1_y = NumericProperty(0)
    # forward sensor vector
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    # initializing x of left sensor
    sensor2_x = NumericProperty(0)
    # initializing y of left sensor
    sensor2_y = NumericProperty(0)
    # left sensor vector
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    # initializing x of right sensor
    sensor3_x = NumericProperty(0)
    # initializing y of right sensor
    sensor3_y = NumericProperty(0)
    # right sensor vector
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    # initializing signal recieved from f-sensor
    signal1 = NumericProperty(0)
    # initializing signal recieved from l-sensor
    signal2 = NumericProperty(0)
    # initializing signal recieved from r-sensor
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # update position of car according to its last position and speed
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # updating position of sensors
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos

        # getting signals recieved from sensors => density of wall or sand aside it
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

        # checking if any sensor has detected full density of wall or sand
        if self.sensor1_x>length-10 or self.sensor1_x<10 or self.sensor1_y>width-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>length-10 or self.sensor2_x<10 or self.sensor2_y>width-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>length-10 or self.sensor3_x<10 or self.sensor3_y>width-10 or self.sensor3_y<10:
            self.signal3 = 1.

# sensors
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# the main class
class Game(Widget):

    # getting objects from kivy file
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        # car starts in the center of screen going right with speed of 6
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):# update function for updating everything in new
        
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global length
        global width

        length = self.width
        width = self.height

        # to initialize map only once
        if first_update:
            init()

        diffrence_x = goal_x - self.car.x
        diffrence_y = goal_y - self.car.y
        # setting orientation of the agent according to goal
        orientation = Vector(*self.car.velocity).angle((diffrence_x,diffrence_y))/180.
        # our input state according to sensors and orientation
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # getting action from ai
        action = brain.update(last_reward, last_signal)
        # appending new score to **score window**
        scores.append(brain.score())
        rotation = action2rotation[action]
        # moving car according to rotation
        self.car.move(rotation)
        # setting new distance
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        # updating sensors new position
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:# changing speed when going into walls or sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            # getting a very bad reward => -1
            last_reward = -0.5
        else: 
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            # normal move . reward => -0.2
            last_reward = -0.2
            if distance < last_distance:
                # if in correct direction get a little positive reward
                last_reward = 0.3

        if self.car.x < 10:# if car goes to left border of screen
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:# if car goes to right side border of screen
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:# if car goes to top border of screen
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:# if car goes to bottom border of screen
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        # updating last distance to goal
        last_distance = distance


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.6,0.5,0.1)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((length,width))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

if __name__ == '__main__':
    CarApp().run()
