#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime 


# In[6]:



audio_file='UrbanSound8K/17973-2-0-32.wav'

librosa_ses_verisi, librosa_ornek_orani = librosa.load(audio_file)


# In[7]:



print(librosa_ses_verisi)


# In[8]:



plt.figure(figsize=(9, 6))
plt.plot(librosa_ses_verisi)
plt.show()


# In[10]:



from scipy.io import wavfile as wav
wave_ornek_orani, wave_ses_verisi = wav.read(audio_file)


# In[11]:


wave_ses_verisi


# In[12]:



plt.figure(figsize=(12, 4))
plt.plot(wave_ses_verisi)
plt.show()


# # FEATURE EXTRACTİON

# In[13]:


mfccs = librosa.feature.mfcc(y=librosa_ses_verisi, sr=librosa_ornek_orani, n_mfcc=40)   
print(mfccs.shape)


# In[14]:


mfccs # dijital degerler


# In[15]:


audio_dataset='UrbanSound8K/audio/'
metadata=pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head()


# In[16]:


def features_extractor(filename):
    ses, ornek_orani = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=ses, sr=ornek_orani, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

# dosyalarin hepsine feature extraction uygulamamiz lazim


# In[17]:



extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# In[18]:



extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[19]:


# Veri kümesini dependent ve independent olarak ayiralim
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[20]:


X.shape


# In[21]:


X


# In[22]:


y


# In[23]:


y.shape


# In[24]:


# one-hot encoding

# 1 0 0 0 0 0 0 0 0 0 => air_conditioner
# 0 1 0 0 0 0 0 0 0 0 => car_horn
# 0 0 1 0 0 0 0 0 0 0 => children_playing
# 0 0 0 1 0 0 0 0 0 0 => dog_bark
# ...
# 0 0 0 0 0 0 0 0 0 1 => street_music

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[25]:


y


# In[26]:


y[0]


# In[27]:



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[28]:



X_train


# In[29]:


y


# In[30]:


X_train.shape


# In[31]:


X_test.shape


# In[32]:


y_train.shape


# In[33]:


y_test.shape


# # CNN

# In[34]:



num_labels = 10


# In[35]:




model=Sequential()

model.add(Dense(125,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[36]:


model.summary()


# In[37]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[38]:



epochscount = 300
num_batch_size = 32

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=epochscount, validation_data=(X_test, y_test), verbose=1)


# In[39]:


validation_test_set_accuracy = model.evaluate(X_test,y_test,verbose=0)
print(validation_test_set_accuracy[1])


# In[40]:


X_test[1]


# In[43]:


predict_x=model.predict(X_test) 


# ### Test 1 Polis siren sesi

# In[67]:


filename="UrbanSound8K/PoliceSiren.wav"
ses_verisi, ornek_orani = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=ses_verisi, sr=ornek_orani, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


# In[68]:


print(mfccs_scaled_features)


# In[69]:


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)


# In[70]:


mfccs_scaled_features.shape


# In[71]:


print(mfccs_scaled_features)


# In[72]:


print(mfccs_scaled_features.shape)


# In[73]:


result_array = model.predict(mfccs_scaled_features)


# In[74]:


result_array 


# In[75]:


result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

result = np.argmax(result_array[0])
print(result_classes[result]) 


# ### Test 2 

# In[76]:


filename="UrbanSound8K/101415-3-0-2.wav"
ses_verisi, ornek_orani = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=ses_verisi, sr=ornek_orani, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


# In[77]:


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)


# In[78]:


result_array = model.predict(mfccs_scaled_features)


# In[79]:


result_array 


# In[80]:


result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

result = np.argmax(result_array[0])
print(result_classes[result]) 


# ### Test 3 

# In[81]:


filename="UrbanSound8K/101415-3-0-2.wav"
ses_verisi, ornek_orani = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=ses_verisi, sr=ornek_orani, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


# In[82]:


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)


# In[83]:


result_array = model.predict(mfccs_scaled_features)


# In[84]:


result_array 


# In[85]:


result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

result = np.argmax(result_array[0])
print(result_classes[result]) 


# In[ ]:




