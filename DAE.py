# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 19:59:05 2021

@author: Coolnerdn from ASUS
"""
import scipy.io as scio
import numpy as np
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

def load_dataset():
    mylist = []
    for i in range(0,6):
        print('collecting gesture',i+1,'...')
        data = scio.loadmat('DFS_2\g' + str(i+1) + '_r2_denoise_2.mat')
        temp = data['final_doppler']
        temp = temp[:100]
        print(temp.shape)
        mylist.append(temp)
    X = np.array(mylist)
    x_per_person = 100
    classes = 6
    X = X.reshape(x_per_person*classes, 1, 121, 1000)
    X = X[:,:,:120,:]
    print(X.shape)
    X = X.transpose((0,3,2,1))
    print('Target DataSet: ', X.shape)
    return X
    
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 40:
        lr *= 1e-4
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

import tensorflow as tf

def AE_model(input_shape):
    input_image=Input(shape=input_shape)
    encoder=Conv2D(32,(4,4),padding='same',activation='relu')(input_image)
    encoder=MaxPooling2D((2,2))(encoder)
    encoder=Conv2D(16,(4,4),padding='same',activation='relu')(encoder)
    encoder=MaxPooling2D((2,2))(encoder)
    encoder=Conv2D(8,(4,4),padding='same', activation='relu')(encoder)
    encoder_out=MaxPooling2D((2,2),name = 'features')(encoder)
    
    decoder=UpSampling2D((2,2))(encoder_out)
    decoder=Conv2D(8,(4,4),padding='same',activation='relu')(decoder)
    decoder=UpSampling2D((2,2))(decoder)
    decoder=Conv2D(16,(4,4),padding='same',activation='relu')(decoder)
    decoder=UpSampling2D((2,2))(decoder)
    decoder=Conv2D(32,(4,4),padding='same',activation='relu')(decoder)
    decoder_out=Conv2D(1, (4, 4), padding='same',activation='sigmoid')(decoder)
                                                                    
    autoencoder=Model(input_image,decoder_out)

    return autoencoder


def myloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
    
x_train = load_dataset()
input_shape = x_train.shape[1:]

x_train_noisy = x_train + np.random.normal(0, (np.max(x_train) - np.min(x_train)) / 2, np.shape(x_train))

x_train =(x_train-x_train.min())/(x_train.max()-x_train.min()) 
x_train_noisy =(x_train_noisy-x_train_noisy.min())/(x_train_noisy.max()-x_train_noisy.min()) 

# Training parameters
batch_size = 30
epochs = 30


model = AE_model(input_shape=input_shape)

model.compile( loss='binary_crossentropy', #loss='mse',
              optimizer=Adam(lr=0.001))#lr_schedule(0)

model.summary()

model.fit(x_train_noisy,x_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_train_noisy,x_train),
              shuffle=True)

model.save_weights('wnDAE_model_weights_r2_DFS2.h5')
