import numpy as np
import struct
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import random


random.seed(1)
S = random.sample(range(60000),50000)


train_x = open("train-images.idx3-ubyte","rb")
train_y = open("train-labels.idx1-ubyte","rb")

img = train_x.read() 
label = train_y.read() 

index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , img , index)

index += struct.calcsize('>IIII')
image = [] 
for i in range(numImages):
    im = struct.unpack_from(">784B",img,index)
    index += struct.calcsize(">784B")
    im = np.array(im)
    im = np.reshape(im,(28,28))
    image.append(im)
image = np.array(image)

# =============================================================================
# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(im , cmap='gray')
# plt.show()
# =============================================================================

index = 0
magic_L, numLabel  = struct.unpack_from('>II' , label , index)
index += struct.calcsize('>II')

Label = np.empty(numLabel) 
for i in range(numLabel):
    Label[i] = struct.unpack_from(">B", label, index)[0]
    index += struct.calcsize(">B")
    

image_train = image[S,:,:]
Label_train = Label[S]

Label = keras.utils.to_categorical(Label,10)
Label_train = keras.utils.to_categorical(Label_train,10)

# =============================================================================

model = keras.Sequential(
    [
        keras.Input(shape=(28,28)),
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
hist = model.fit(image_train,Label_train,epochs=10)

regular_accuracy = hist.history["accuracy"] 
regular_loss =  hist.history["loss"]

# =============================================================================

model = keras.Sequential(
    [
        keras.Input(shape=(28,28)),
        layers.BatchNormalization(),
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
hist = model.fit(image_train, Label_train,epochs=10)


batch_accuracy = hist.history["accuracy"] 
batch_loss =  hist.history["loss"]
# =============================================================================
Epoch = list(range(10))

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(Epoch,batch_accuracy,'s-',color = 'r', label="batch")
plt.plot(Epoch,regular_accuracy,'s-',color = 'b', label="regular")
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
plt.ylabel("accuracy", fontsize=30, labelpad = 20)
plt.legend(loc = "best", fontsize=20)
plt.show()

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(Epoch,batch_loss,'s-',color = 'r', label="batch")
plt.plot(Epoch,regular_loss,'s-',color = 'b', label="regular")
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
plt.ylabel("cost", fontsize=30, labelpad = 20)
plt.legend(loc = "best", fontsize=20)
plt.show()

# =============================================================================
#第二題
# =============================================================================

model2 = keras.Sequential(
     [
        keras.Input(shape=(28,28)),
        layers.BatchNormalization(),
        layers.Dense(100, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(100, activation="relu"),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
     ]
)

model2.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
hist2 = model2.fit(image,Label, batch_size=500, epochs=100,validation_split=1/6)

train_accuracy = hist2.history["accuracy"]
val_accuracy = hist2.history["val_accuracy"]
train_loss = hist2.history["loss"]
val_loss = hist2.history["val_loss"]

# =============================================================================

Epoch = list(range(100))

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(Epoch,train_accuracy,color = 'r', label="train")
plt.plot(Epoch,val_accuracy,color = 'b', label="validation")
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
plt.ylabel("accuracy", fontsize=30, labelpad = 20)
plt.legend(loc = "best", fontsize=20)
plt.show()

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(Epoch,train_loss,color = 'r', label="train")
plt.plot(Epoch,val_loss,color = 'b', label="validation")
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
plt.ylabel("cost", fontsize=30, labelpad = 20)
plt.legend(loc = "best", fontsize=20)
plt.show()













