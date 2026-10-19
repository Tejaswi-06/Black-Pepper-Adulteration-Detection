from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
from tkinter import filedialog

import cv2
from PIL import Image,ImageTk
import imutils
import numpy as np
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense,Activation,Flatten, Dropout




root = Tk()
root.title('DETECTION OF BLACKPEPPER ADULTERATION')
root.geometry('1920x1080')
style = Style()


def Input_Image():
    global resized_image
    global file_path
    root.filename =  filedialog.askopenfilename(parent=root,initialdir = "/",title = "choose your file",filetypes = (("all files",".*"),("png files",".png")))
    file_path = root.filename
    print(root.filename)
    img = cv2.imread(root.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img,(350,300))
   
    image = Image.fromarray(resized_image)
    image = ImageTk.PhotoImage(image)
    panelA = Label(image=image)
    panelA.image = image
    panelA.place(x=40 ,y=120)
   
def preprocessing_image():
    global resized_image
    global thresh
    gray= cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    image = Image.fromarray(thresh)
    image = ImageTk.PhotoImage(image)
    panelA = Label(image=image)
    panelA.image = image
    panelA.place(x=490 ,y=120)
   

def segment_image():
    global resized_image
    global thresh
    global res_red
    hsv = cv2.cvtColor(resized_image,cv2.COLOR_BGR2HSV)
    Z = hsv.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    khsv   = center[label.flatten()]
    khsv   = khsv.reshape((resized_image.shape))
    res_red = cv2.bitwise_and(resized_image,resized_image, mask=thresh)
 
    image = Image.fromarray(res_red)
    image = ImageTk.PhotoImage(image)
    panelA = Label(image=image)
    panelA.image = image
    panelA.place(x=950 ,y=120)

def seed_detection_cnn():
    global file_path
    model = load_model('cnntrain.h5')
    clas = {1:'noncancerous',2:'melanoma',3:'nevus'}
    image = Image.open(file_path).convert("RGB")
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    classes = np.argmax(pred, axis=-1)
    sign = clas[classes+1]
    print(sign)
    e1.delete(0,'end')
    e1.insert(5,sign) 

def seed_detection_knn():
    global file_path
    model = load_model('rnntrain.h5')
    clas = {1:'noncancerous',2:'melanoma',3:'nevus'}
    image = Image.open(file_path).convert("RGB")
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image.reshape(image.shape[0], -1, 1) 
    pred = model.predict([image])[0]
    classes = np.argmax(pred, axis=-1)
    sign = clas[classes+1]
    print(sign) 
    e1.delete(0,'end')
    e1.insert(5,sign)
    


def seed_Detection():
    g1=combo.get()
    if(g1=="KNN"):
        seed_detection_knn()
    if(g1=="CNN"):
        seed_detection_cnn()


def Train_CNN():
    data = []
    labels = []
    classes = 3
    cur_path = os.getcwd()
    for i in range(classes):
        path = os.path.join(cur_path,'train',str(i))
        images = os.listdir(path)
        for a in images:
            try:
                print(path + '\\'+ a)
                image = Image.open(path + '\\'+ a).convert("RGB")
                print(image.format, image.size, image.mode)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    #Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    
    #Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=1)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #Converting the labels into one hot encoding
    y_train = to_categorical(y_train,3)
    y_test = to_categorical(y_test,3)

    #Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(3, activation='softmax'))

    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 25
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    model.save("cnntrain.h5")


def Train_KNN():
    batch_size = 32
    classes = 3
    epochs = 25
    data = []
    labels = []
    hidden_units = 10

    learning_rate = 1e-6
    clip_norm = 1.0
    cur_path = os.getcwd()

    #Retrieving the images and their labels 
    for i in range(classes):
        path = os.path.join(cur_path,'train',str(i))
        print(path)
        images = os.listdir(path)

        for a in images:
            try:
                print(path + '\\'+ a)
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    #Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)
    #Splitting training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


    x_train = x_train.reshape(x_train.shape[0], -1, 1)
    x_test = x_test.reshape(x_test.shape[0], -1, 1)
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)


    model = Sequential()
    model.add(SimpleRNN(hidden_units,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu',
                        input_shape=x_train.shape[1:]))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    model.save("rnntrain.h5")








menubar = Menu(root)

test = Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Test', menu = test)
test.add_command(label ='Input image', command = Input_Image)
test.add_command(label ='Preprocessing', command = preprocessing_image)
test.add_command(label ='Segmentation', command = segment_image)  

train = Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Train', menu = train)
train.add_command(label ='Train_KNN', command = Train_KNN)
train.add_command(label ='Train_CNN', command = Train_CNN)


root.config(menu = menubar)

Label(root, text = 'DETECTION OF BLACKPEPPER ADULTERATION',foreground="red", font =('Verdana', 25)).pack(side = TOP, pady = 10)
 
w = Canvas(root, width=500, height=600)
label = Label(root, text ="INPUT IMAGE",foreground="red", font =('Verdana', 12))
label.place(x=160 ,y=90)
w.place(x=40 ,y=120)
w.create_rectangle(10, 10, 350, 300,outline='grey', fill="lightgray")

x = Canvas(root, width=500, height=600)
label = Label(root, text ="PREPROCESSING",foreground="red", font =('Verdana', 12))
label.place(x=600 ,y=90)
x.place(x=490 ,y=120)
x.create_rectangle(10, 10, 350, 300,outline='grey', fill="lightgray")

y = Canvas(root, width=500, height=600)
label = Label(root, text ="SEGMENTATION",foreground="red", font =('Verdana', 12))
label.place(x=1070 ,y=90)
y.place(x=950 ,y=120)
y.create_rectangle(10, 10, 350, 300,outline='grey', fill="lightgray")

z = Canvas(root, width=1260, height=350)
label = Label(root, text ="OUTPUT",foreground="red", font =('Verdana', 12))
label.place(x=660 ,y=430)
z.place(x=40 ,y=450)
z.create_rectangle(10, 10,1260, 200,outline='black',fill="lightgray",width=1)

combo = ttk.Combobox(root,font =('Verdana', 12),foreground = 'red',state="readonly",values=["KNN","CNN"])
label = Label(root,text ="SELECT ALGORITHM",background="lightgray",foreground="red",font =('Verdana', 10))
label.place(x=165 ,y=480)
combo.place(height=30,width=230,x=121, y=500)
combo.current(0)

style.configure('W.TButton',font =('Verdana', 12),foreground = 'red',background='lightgray')

b1=Button(root, text ="ADULTERATION DETECTION",style='W.TButton',command = seed_Detection)
b1.place(height=80, width=233,x=120 ,y=540)

label = Label(root,text ="DETECTED SEEDS",background="lightgray",foreground="red",font =('Verdana', 10))
label.place(x=620 ,y=480)
e1 = Entry(root,font =('Verdana', 12),foreground = 'red',justify = CENTER)
e1.place(height=115, width=300,x=550 ,y=500)

label = Label(root,text ="ACCURACY IN %",background="lightgray",foreground="red",font =('Verdana', 10))
label.place(x=1000 ,y=480)
e2 = Entry(root,font =('Verdana', 12),foreground = 'red',justify = CENTER)
e2.place(height=115, width=300,x=930 ,y=500)

mainloop()
