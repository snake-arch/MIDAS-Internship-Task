import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf

# Code to ensure GPU is used if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

# Path to the dataset

path = r'mnistTask'


# Importing images to memmeory and resizing them to 75x75 pixels so that they don't overload the GPU/CPU
myList = os.listdir(path)
noOfClasses = len(myList)
print("Total classes="+str(len(myList)))
classNo=[]
images=[]
print("Importing classes...")

# Images are appended to the 'images' array and labels to the 'labels' array
for x in myList:
    myimgList=os.listdir(path+"/"+str(x))
    for y in myimgList:
        curimg=cv2.imread(path+"/"+str(x)+"/"+y)
        curimg=cv2.resize(curimg,(75,75))
        images.append(curimg)
        classNo.append(x)
    print(x,end=" ")
#print("Total images found "+str(len(images)))
#for i in range(0,noOfClasses):
#print("no of "+str(i)+" images "+str(classNo.count(i)))

#Converting the images and labels into numpy array
images=np.array(images)
classNo=np.array(classNo)
print(len(images))
print(len(classNo))

# Splitting the dataset into train,test and validation
X_train,X_test,y_train,y_test= train_test_split(images,classNo,test_size=0.2)
X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=0.2)

# Normalizing the image so that the pixel values range from 0-1 only
# Colour is not required as the colour does not define a no or its not a feature thus converted to gray
def Preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

#Using the map function to apply preprocessing at once
X_train=np.array(list(map(Preprocessing,X_train)))
X_test=np.array(list(map(Preprocessing,X_test)))
X_validate=np.array(list(map(Preprocessing,X_validate)))

# Reshaping the array to ensure that multiple channels are dropped to '1'
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validate=X_validate.reshape(X_validate.shape[0],X_validate.shape[1],X_validate.shape[2],1)

#shifts the image here and there to generate  randomness in data
# important so that the model learns non-mainstream scenarios where the object could be present
#datagen= ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)

#fiting the data to the augmentor
#datagen.fit(X_train)

#coverting output to categories # used label encoding as the number of categories is quite large as one-hot encoding can lead to high memory consumption
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
y_validate = label_encoder.fit_transform(y_validate)
y_train=to_categorical(y_train,noOfClasses)
y_test=to_categorical(y_test,noOfClasses)
y_validate=to_categorical(y_validate,noOfClasses)

# model checkpointing
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5" #each file with lower val loss will be saved
checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min",save_best_only=True, verbose=1)
# monitors can be val_accuracy too mode decides what to go towards min or max

#Defining the callbacks
callbacks = [checkpoint]

# making a model object
model=load_model('weights-improvement-129-0.41.hdf5')
model.add(Dense(units = 10, name = "dense_last", activation = 'softmax'))# Batch size can be less here becuase the dataset is small
batchSizeValue=64

# High epoch value so that it can be examined when overfitting happens
epochVal=15

# Fitting the model with shuffle value one
history=model.fit(X_train,y_train,
                    epochs=epochVal,
                    validation_data=(X_validate,y_validate),callbacks=callbacks,shuffle=1)


# Using matplotlib to plot the training & validation loss/accuracy overtime
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score=model.evaluate(X_test,y_test,verbose=0)
print('Test Score=',score[0])
print('Test Accuracy=',score[1])