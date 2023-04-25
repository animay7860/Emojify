import tkinter as tk
from tkinter import *
# import cv2
from PIL import Image
from PIL import ImageTk
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading
emotion_model = Sequential()#to extract the features in model
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

# emotion dictionary contains the emotions present in the dataset
em_dict = {0: "   Angry   ", 1: " Disgusted ", 2: "  Fearful  ", 3: "   Happy   ", 4: "Neutral", 5: "    Sad    ", 6: "Surprised"}
cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist={0 :'D:/Project/Emojify/src/emojis/angry.png', 1:'D:/Project/Emojify/src/emojis/disgusted.png', 2:'D:/Project/Emojify/src/emojis/fearful.png',3:  'D:/Project/Emojify/src/emojis/happy.png',4:'D:/Project/Emojify/src/emojis/neutral.png',5:'D:/Project/Emojify/src/emojis/sad.png',6:'D:/Project/Emojify/src/emojis/surprised.png'}
global last_frame1    #emoji dictionary is created with images for every emotion present ion dataset                               
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
def show_vid():    #to open the camera and to record video
    cap1 = cv2.VideoCapture(0)      #it starts capturing                          
    if not cap1.isOpened():  #if camera is not open 
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))#to resize the image frame
    bound_box = cv2.CascadeClassifier('C:/Users/ANIMAY PRAKASH/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')#it will detect the face in the video and bound it with a rectangular box
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)#to color the frame
    n_faces = bound_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in n_faces: #for n different faces of a video
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_frame = gray_frame[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)#crop the image and save only emotion contating face
        prediction = emotion_model.predict(crop_img)#predict the emotion from the cropped image
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, em_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex#store the emotion found in image from emotion dictionary
    if flag1 is None:#if webcam is disabled
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB) #to store the image   
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
        root.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])#to store the emoji with respect to the emotion
    pic2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=em_dict[show_text[0]],font=('arial',45,'bold'))#to configure image and text
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_vid2)

if __name__ == '__main__':
    root=tk.Tk()  
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    root.title("Photo To Emoji")           
    root.geometry("1400x900+100+10")
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)

    threading.Thread(target=show_vid).start()
    threading.Thread(target=show_vid2).start()
    # show_vid()#function calling to record video
    # show_vid2()#function calling to generate emoji from recorded video
    root.mainloop()