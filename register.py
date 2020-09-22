import tkinter as tk
from tkinter import Message, Text
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as font
import pyrebase
import smtplib



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


window = tk.Tk()
window.title("Kathmandu Bernhardt College")
window.configure(background='gray')
window.geometry('1280x670')


lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=2, font=('times', 30, 'italic bold')) 
lbl.place(x=100, y=20)

reg = tk.Label(window, text="Register", width=57, fg="black", bg="white",height=2, font=('times', 15, ' bold')) 
reg.place(x=340, y=130)

e1= tk.Label(window, text="Enter Email", width=15, fg="black", bg="white", height=2, font=('times', 15, ' bold')) 
e1.place(x=340, y=200)

e2 = tk.Entry(window, width=40, bg="white", fg="black", font=('times', 15, ' bold '))
e2.place(x=550, y=210)

p1 = tk.Label(window, text="Password", width=15, fg="black", bg="white", height=2, font=('times', 15, ' bold')) 
p1.place(x=340, y=300)

p2 = tk.Entry(window, width=40, show="*",bg="white", fg="black", font=('times', 15, ' bold '))
p2.place(x=550, y=310)

p3 = tk.Label(window, text="Confirm Password", width=15, fg="black", bg="white", height=2, font=('times', 15, ' bold')) 
p3.place(x=340, y=400)

p4 = tk.Entry(window, width=40, show="*",bg="white", fg="black", font=('times', 15, ' bold '))
p4.place(x=550, y=410)

def registerdata():
    emaildata=e2.get()
    passdata=p2.get()
    confirmpass=p4.get()
    
    try:
        if(passdata == confirmpass):
            a=auth.create_user_with_email_and_password(emaildata,passdata)
            window = tk.Tk()
            window.title("Kathmandu Bernhardt College")
            window.configure(background='gray')
            window.geometry('1280x670')
            sender="gaire.avinash@gmail.com"
            receiver=emaildata
            passowrd="email2me"
            message="You are registered Successfully"
            server=smtplib.SMTP('smtp.gmail.com',587)
            server.starttls()
            server.login(sender,passowrd)
            print("hello")
            server.sendmail(sender,receiver,message)
            print("emailsent :",receiver)

            s1 = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=2, font=('times', 30, 'italic bold')) 
            s1.place(x=100, y=20)
            s2 = tk.Label(window, text="Successfully registered,an email has been sent", bg="white" , fg="black" , width=80 , height=4, font=('times', 20, 'italic bold')) 
            s2.place(x=50, y=200)
            quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
            quitWindow.place(x=400, y=400)
        
            
    except:
        window = tk.Tk()
        window.title("Kathmandu Bernhardt College")
        window.configure(background='gray')
        window.geometry('1280x670')
        lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=2, font=('times', 30, 'italic bold')) 
        lbl.place(x=100, y=20)

        er1 = tk.Label(window, text="Please Enter Valid Email and Password", bg="white" , fg="black" , width=80 , height=4, font=('times', 20, 'italic bold')) 
        er1.place(x=50, y=200)

        quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=300, y=400)
        
        window.mainloop()
            
    



register = tk.Button(window, text="Register", fg="black",command=registerdata ,bg="white", width=10, height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
register.place(x=500, y=500)


quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=10, height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=700, y=500)

window.mainloop()