import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
####################################
path = r'mnistTask'


####################################
myList = os.listdir(path)
noOfClasses = len(myList)
print("Total classes="+str(len(myList)))
classNo=[]
images=[]
print("Importing classes...")
for x in range(0,noOfClasses):
    myimgList=os.listdir(path+"/"+str(x))
    for y in myimgList:
        curimg=cv2.imread(path+"/"+str(x)+"/"+y)
        curimg=cv2.resize(curimg,(75,75))
        images.append(curimg)
        classNo.append(x)
    print(x,end=" ")
#print("Total images found "+str(len(images)))
#for i in range(0,noOfClasses):
#    print("no of "+str(i)+" images "+str(classNo.count(i)))
images=np.array(images)
classNo=np.array(classNo)
print(len(images))
print(len(classNo))
X_train,X_test,y_train,y_test= train_test_split(images,classNo,test_size=0.2)
X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=0.2)


def Preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

X_train=np.array(list(map(Preprocessing,X_train)))
X_test=np.array(list(map(Preprocessing,X_test)))
X_validate=np.array(list(map(Preprocessing,X_validate)))

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validate=X_validate.reshape(X_validate.shape[0],X_validate.shape[1],X_validate.shape[2],1)

#shifts the image here and there to generate  randomness in data
datagen= ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
#fit the data
datagen.fit(X_train)
#coverting output to categories
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
y_validate = label_encoder.fit_transform(y_validate)
y_train=to_categorical(y_train,noOfClasses)
y_test=to_categorical(y_test,noOfClasses)
y_validate=to_categorical(y_validate,noOfClasses)

def myModel():
    noOfFilters=60
    sizeOfFilter=(5,5)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter,input_shape=(75,75,1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5" #each file with lower val loss will be saved
checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min",save_best_only=True, verbose=1) # monitors
# can be val_accuracy too mode decides what to go towards min or max


callbacks = [checkpoint]
model=myModel()
print(model.summary())
batchSizeValue=64
epochVal=15
#stepsPerEpoch=1000
history=model.fit_generator(datagen.flow(X_train,y_train,batch_size=batchSizeValue),
                    epochs=epochVal,
                    validation_data=(X_validate,y_validate),callbacks=callbacks,shuffle=1)


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
