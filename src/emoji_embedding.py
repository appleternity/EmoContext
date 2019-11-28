import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
import numpy as np
import json
import ZipFile
import cv2

#reading the json file content
file_List =[]
y_List=[]
with open('DataInfo.json') as json_file:
    data = json.load(json_file)
    for a in data:
      file_List.append(a[0]);
      y_List.append(np.array(a[1]));

# print(file_List)

#Extracting necessary emoji pictures based on Dataset

with ZipFile("Emoji_Downloaded.zip", 'r') as zip: 
    zipObj = ZipFile('Train_Data.zip', 'w')
    for i in file_List:
      zipObj.write(zip.extract('Emoji_Downloaded/'+str(i)+".png"))
    zipObj.close();
    # print(zip.namelist())
        
with ZipFile('Train_Data.zip', 'r') as zipnewfile: 
    # printing all the contents of the zip file 
    zipnewfile.printdir() 

#Training Images Created

X_List = []
def generateTrainingData():
  with ZipFile("Train_Data.zip", 'r') as zip: 
    for name in zip.namelist():
      name = "'"+name+"'"
      filepath = "/".join(name.strip("/").split('/')[1:])
      filepath=filepath[:-1]
      img_array = cv2.imread(filepath)
      img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) #RGBtoBGR(For Loading Exact Images)
      X_List.append([img_array])
      # print(img_array.shape)
      
      # break;
generateTrainingData()

plt.imshow(X_List[2][0])

plt.show()
print(y_List[2])

#converting the labels and imagedata into numpyarray from the list
X_List = np.array(X_List).reshape(-1, 128, 128, 3)
y_List = np.vstack(y_List) 
# print(X_List.shape)
# print(y_List.shape)

# print(y_List)
# type(y_List)

#Normalizing the data

X_List = X_List/255.0
# print(X_List)

#Using Keras for implementing the CNN for extracting information


model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop", 
              metrics=['accuracy'])

model.fit(X_List, y_List, epochs= 15, batch_size=64, validation_split=0.2)
