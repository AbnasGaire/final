import tkinter as tk
from tkinter import Message, Text
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as font
import pyrebase
def configuration():
  firebaseConfig = {
      'apiKey': "AIzaSyDi6cFaZK-kXtN3vD3_7Cb5VHBOlBYaHWg",
      'authDomain': "facerecognition-2a543.firebaseapp.com",
      'databaseURL': "https://facerecognition-2a543.firebaseio.com",
      'projectId': "facerecognition-2a543",
      'storageBucket': "facerecognition-2a543.appspot.com",
      'messagingSenderId': "825290085951",
      'appId': "1:825290085951:web:54794ac44b415bf8d88a8a",
      'measurementId': "G-2KLXP1FL1Z"
    }

  firebase=pyrebase.initialize_app(firebaseConfig)
  auth=firebase.auth()



   




  