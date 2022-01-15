環境
=======
google colaboratory
Hardware accelerator：GPU

流程
=======
1.匯入函式庫\
2.讀取圖片、標籤、名稱\
3.分割測試集、訓練集\
4.建立模型\
5.訓練模型\
6.資料視覺化\
7.使用模型預測資料\
8.輸出預測資料

匯入函式庫
=======
```python
#import
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import glob
import h5py
import sys
import cv2
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout,LeakyReLU,BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras import datasets,layers,models,callbacks,metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.models import load_model,Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import utils
from random import shuffle
```

讀取圖片、標籤、名稱
=======
```python
classNames=[]
cntClass=0
trainPath='/content/data/theSimpsons-train/train/'

workIn=os.getcwd()
os.chdir (trainPath)

for d in os.listdir('.'):
    if os.path.isdir(trainPath):
        classNames.append(d)
        cntClass+=1

os.chdir (trainPath)

print(classNames)
print(cntClass)
#=========================================================
def readPicLabels(path,i):
    for f in os.listdir(path):
        absPath=os.path.abspath(os.path.join(path,f))
        if os.path.isdir(absPath):
            i+=1
            temp=os.path.split(absPath)[-1]
            readPicLabels(absPath, i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i) +'[%s%%]' % (i*100/amount)+temp)
        if f.endswith('.jpg'):
            tmpImg=cv2.imread(absPath)
            tmpImg=cv2.cvtColor(tmpImg,cv2.COLOR_BGR2RGB)
            image = cv2.resize(tmpImg, (imgW, imgH))
            images.append(image)
            labels.append(i-1)
    return images,labels

def readPic(path):
    img,labels=readPicLabels(path,i=0)########,name
    img=np.array(img,dtype=np.float32)/255
    labels=utils.to_categorical(labels,num_classes=50)
    return img,labels
#=========================================================
imgs,lbls=readPic(trainPath)
```

分割測試集、訓練集
=======
```python
Xtrain,Xtest,Ytrain,Ytest=train_test_split(imgs,lbls,test_size=0.15)
```

建立模型
=======
![image](https://github.com/MachineLearningNTUT/classification-NTUB110002016/blob/main/HW2/Pics/model.png)

訓練模型
=======
```python
batchSize = 64
epochs = 50
stepsPerEpoch=1000
# run過5+28+50


#REF:https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36
datagen = ImageDataGenerator(
  rotation_range=0.1, # 隨機旋轉範圍內的圖像
  width_shift_range=0.1, # 隨機水平移動圖像
  height_shift_range=0.1, # 隨機垂直移動圖像
  horizontal_flip=True) # 隨機翻轉圖片

datagen.fit(Xtrain)

history=model.fit_generator(datagen.flow(Xtrain, Ytrain, batch_size=batchSize),
                 epochs=epochs,steps_per_epoch=stepsPerEpoch,
                 validation_data=(Xtest,Ytest),
                 shuffle=True,
                 callbacks=[ModelCheckpoint('model.h5',save_best_only=True)])
```

資料視覺化
=======
![image](https://github.com/MachineLearningNTUT/classification-NTUB110002016/blob/main/HW2/Pics/plotTrainHistory.png)

使用模型預測資料
=======
```python
testPath='/content/data/theSimpsons-test/test/'

def readImages(path):
    images = []
    for i in range(10791):
        image = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'), (imgW, imgH))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(image)
    images = np.array(images, dtype=np.float32)/255
    return images

readTestImage=readImages(testPath)
print(readTestImage.shape)
#=========================================================
model=load_model('model.h5')
pred=model.predict(readTestImage)
pred=np.argmax(pred,axis=1)
```

輸出預測資料
=======
```python
with open('testVer9.csv','w') as f:
    f.write('id,character\n')
    for i in range(10791):
        f.write(str(i+1)+","+str(classNames[pred[i]])+"\n")
```

結果分析
=======
![image](https://github.com/MachineLearningNTUT/classification-NTUB110002016/blob/main/HW2/Pics/my%20submission.png)

心得
=======
當初想說訓練時間會超乎想像，就利用anaconda去做，建立環境時遇到許多問題，但我都一一解決了，但score是在0.02左右，最後我還是放棄利用Colab去完成這次的作業，在Public Leaderboard最佳也只拿到0.85542，但這大大的變化只是因為我增加了COLOR_BGR2RGB以及將resize的值縮小而已。
