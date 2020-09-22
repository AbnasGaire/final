import tkinter as tk
from tkinter import Message, Text
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as font
import pyrebase



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

e1 = tk.Label(window, text="Enter Email", width=20 , height=2 , fg="black" , bg="white", font=('times', 15, ' bold ') ) 
e1.place(x=200, y=200)

e2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
e2.place(x=550, y=215)

p1 = tk.Label(window, text="Enter Password", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
p1.place(x=200, y=300)

p2 = tk.Entry(window, width=20, bg="white",show="*", fg="black", font=('times', 15, ' bold ')  )
p2.place(x=550, y=315)





def loginprocess():
    email=e2.get()
    password=p2.get()

    try:
        auth.sign_in_with_email_and_password(email,password)
        window = tk.Tk()
        window.title("Kathmandu Bernhardt College")
        window.configure(background='gray')
        window.geometry('1280x670')

        lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=2, font=('times', 30, 'italic bold')) 
        lbl.place(x=100, y=20)

        lbl1 = tk.Label(window, text="Enter ID", width=20 , height=2 , fg="black" , bg="white", font=('times', 15, ' bold ') ) 
        lbl1.place(x=200, y=200)

        txt1 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        txt1.place(x=550, y=215)

        lbl2 = tk.Label(window, text="Enter Name", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
        lbl2.place(x=200, y=300)

        txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold ')  )
        txt2.place(x=550, y=315)

        lbl4 = tk.Label(window, text="Enter Sem", width=20 , height=2 , fg="black" , bg="white", font=('times', 15, ' bold ') ) 
        lbl4.place(x=200, y=400)

        txt4 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold ')  )
        txt4.place(x=550, y=400)

        lbl3 = tk.Label(window, text="Notification →", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
        lbl3.place(x=200, y=500)

        message = tk.Label(window, text="", bg="white", fg="black", width=30, height=2, font=('times', 15, ' bold ')) 
        message.place(x=550, y=500)
        
        def clearId():
            txt1.delete(0, 'end')

        def clearName():
            txt2.delete(0, 'end')


        def clearSem():
            txt4.delete(0, 'end')

        def isNumber(s):
            try:
                float(s)
                return True
            except ValueError:
                pass

        def takeImages():        
            Id=(txt1.get())
            name=(txt2.get())
            sem=(txt4.get())
            if(isNumber(Id) and name.isalpha() and sem.isalpha()):
                cam = cv2.VideoCapture(0)
                harcascadePath = "haarcascade_frontalface_default.xml"
                detector=cv2.CascadeClassifier(harcascadePath)
                sampleNum=0
                while(True):
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                        sampleNum=sampleNum+1
                        cv2.imwrite("SampleImages\ "+name +"."+sem+"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                        cv2.imshow('Face Detecting',img)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    elif sampleNum>60:
                        break
                cam.release()
                cv2.destroyAllWindows() 
                res = "Images Saved for ID : " + Id +" Name : "+ name+"Sem :"+sem
                row = [Id , name , sem]
                with open('StudentRecord.csv','a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
                message.configure(text= res)
            else:
                if(isNumber(name)):
                    res = "Enter Alphabetical Name"
                    message.configure(text= res)
                if(Id.isalpha()):
                    res = "Enter Numeric Id"
                    message.configure(text= res)
                if(isNumber(sem)):
                    res="Enter Semester in Alphabet"
                    messsage.configure(test=res)
            

        
        clearButton1 = tk.Button(window, text="Clear", command=clearId, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
        clearButton1.place(x=850, y=200)

        clearButton2 = tk.Button(window, text="Clear", command=clearName, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
        clearButton2.place(x=850, y=300)  

        clearButton3 = tk.Button(window, text="Clear", command=clearName, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
        clearButton3.place(x=850, y=400)  

        takeImg = tk.Button(window, text="Take Images", command=takeImages, fg="black", bg="white", width=20, height=3, activebackground = "Green", font=('times', 15, ' bold '))
        takeImg.place(x=200, y=600)


        quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=800, y=600)

        lbl4 = tk.Label(window, text="GROUP NAME :Abinash group", width=80, fg="white", bg="black", font=('times', 15, ' bold')) 
        lbl4.place(x=200, y=650)

        window.mainloop()

    

    except:
        window = tk.Tk()
        window.title("Kathmandu Bernhardt College")
        window.configure(background='gray')
        window.geometry('1280x670')
        lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=2, font=('times', 30, 'italic bold')) 
        lbl.place(x=100, y=20)

        er1 = tk.Label(window, text="Please Enter Valid Email and Password", bg="white" , fg="black" , width=80 , height=4, font=('times', 20, 'italic bold')) 
        er1.place(x=50, y=200)
        
        window.mainloop()


login1= tk.Button(window, text="Login",fg="black", bg="white",command=loginprocess, width=20, height=3, activebackground = "Green", font=('times', 15, ' bold '))
login1.place(x=400, y=450)
window.mainloop()





