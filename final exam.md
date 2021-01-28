# 期末考


## 第一題
載入套件
```python
import numpy as np
import struct       
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import random
```
取得亂數(60000抽50000)

```python
random.seed(1)
S = random.sample(range(60000),50000)
```
讀取mnist資料 : 資料路徑為該電腦預設工作目錄
```python
train_x = open("train-images.idx3-ubyte","rb")
train_y = open("train-labels.idx1-ubyte","rb")
img = train_x.read() 
label = train_y.read() 
```
將資料轉成陣列
```python
#image
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
image = np.array(image) #60000張圖*(28*28)
```
```python
#label
index = 0
magic_L, numLabel  = struct.unpack_from('>II' , label , index)
index += struct.calcsize('>II')

Label = np.empty(numLabel) 
for i in range(numLabel):
    Label[i] = struct.unpack_from(">B", label, index)[0]
    index += struct.calcsize(">B")
#Label:60000*1
```
資料抽樣
```python
image_train = image[S,:,:] #50000張圖*(28*28)
Label_train = Label[S]     #50000*1
```
label 做one-hot-encoding
```python
Label = keras.utils.to_categorical(Label,10) #60000*10
Label_train = keras.utils.to_categorical(Label_train,10)          #50000*10
```
regular 模型
```python
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

model.compile(loss="categorical_crossentropy",optimizer="SGD",    metrics=["accuracy"])
hist = model.fit(image_train,Label_train,epochs=10)
regular_accuracy = hist.history["accuracy"] 
regular_loss =  hist.history["loss"]
```
batch normalzation 模型
```python
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
```
繪圖
```python
Epoch = list(range(10))
#accuracy
plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(Epoch,batch_accuracy,'s-',color = 'r', label="batch")
plt.plot(Epoch,regular_accuracy,'s-',color = 'b', label="regular")
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
plt.ylabel("accuracy", fontsize=30, labelpad = 20)
plt.legend(loc = "best", fontsize=20)
plt.show()
#cost
plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(Epoch,batch_loss,'s-',color = 'r', label="batch")
plt.plot(Epoch,regular_loss,'s-',color = 'b', label="regular")
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
plt.ylabel("cost", fontsize=30, labelpad = 20)
plt.legend(loc = "best", fontsize=20)
plt.show()
```
結果

![Pandao editor.md](https://i.imgur.com/kc9Y0KC.png)

![Pandao editor.md](https://i.imgur.com/ow4VCac.png)

## 第二題

模型
```python
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

model2.compile(loss="categorical_crossentropy",optimizer="SGD",  metrics=["accuracy"])
hist2 = model2.fit(image,Label, batch_size=500, epochs=100,validation_split=1/6)

train_accuracy = hist2.history["accuracy"]
val_accuracy = hist2.history["val_accuracy"]
train_loss = hist2.history["loss"]
val_loss = hist2.history["val_loss"]
```
繪圖
```python
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
```
結果

![Pandao editor.md](https://i.imgur.com/qmycmAL.png)

![Pandao editor.md](https://i.imgur.com/XwlpLwl.png)
