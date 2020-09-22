import tkinter as tk
from tkinter import Message, Text
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as font

window = tk.Tk()
window.title("Kathmandu Bernhardt College")
window.configure(background='gray')
window.geometry('1280x670')

lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=2, font=('times', 30, 'italic bold')) 
lbl.place(x=100, y=20)

lbl3 = tk.Label(window, text="Notification →", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
lbl3.place(x=200, y=200)

message = tk.Label(window, text="", bg="white", fg="black", width=30, height=2, font=('times', 15, ' bold ')) 
message.place(x=550, y=200)
 
def trainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("SampleImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("DataSet\Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[2])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

trainImg = tk.Button(window, text="Train Images", command=trainImages, fg="black", bg="white", width=20, height=3, activebackground = "Green" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=400)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=800, y=400)

lbl3 = tk.Label(window, text="KBC", width=80, fg="white", bg="black", font=('times', 15, ' bold')) 
lbl3.place(x=200, y=620)

window.mainloop()