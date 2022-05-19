import kivy
import cv2
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import ObjectProperty
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App

window = (800,600)

Window.clearcolor = (180/255,224/255,216/255,0.88)
Window.size = window

class IntroScreen(Screen):
    pass

class InfoScreen(Screen):
    pass

class MainScreen(Screen):
    pass

class MotionExam(Screen):
    pass

class ScreenManagement(ScreenManager):
    pass

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)

    #Drawing method to execute at intervals
    def update(self, dt):
        #Load frame
        ret, self.frame = self.capture.read()
        #Convert to Kivy Texture
        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

class BtnTextInput(BoxLayout):
    pass

class uiApp(App):
    def build(self):
        return
    def test(self, widget, *args):
        anim = Animation(text_opacity=0, duration=0.1)
        anim.start(widget)
    def exam(self, widget, *args):
        anim = Animation(my_pos=(window[0] / 2, window[1] / 2), duration=.1)
        anim += Animation(my_color = (1,0,0,1), duration = .1)
        anim += Animation(my_pos=(0, window[1] / 2), duration=1.95)
        for i in range(3):
            anim += Animation(my_pos=(window[0] - 25, window[1] / 2), duration=3.9)
            anim += Animation(my_pos=(0, window[1] / 2), duration=3.9)
        anim += Animation(my_pos=(window[0] / 2, window[1] / 2), duration=1.95)
        anim += Animation(my_color=(1, 0, 0, 0), duration=.5)
        anim.start(widget)

uiApp().run()