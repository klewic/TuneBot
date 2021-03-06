import os
import sys
import csv
import numpy as np
import random
import keras
from keras import layers, regularizers

latentDim = 16  # dimension of random noise data
height = 200
width = 1

###############
## generator ##
###############

generatorInput = keras.Input(shape = (latentDim, ))

x = generatorInput

x = layers.Dropout(0.2)(x)

x = layers.Dense(width * height)(x)
x = layers.LeakyReLU()(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(width * height)(x)
x = layers.LeakyReLU()(x)

x = layers.Reshape((height, width))(x)

x = layers.LSTM(width, return_sequences = True)(x)
x = layers.LeakyReLU()(x)

x = layers.LSTM(width, return_sequences = True)(x)
x = layers.LeakyReLU()(x)

x = layers.LSTM(width, return_sequences = True)(x)
x = layers.LeakyReLU()(x)

x = layers.LSTM(width, return_sequences = True)(x)
x = layers.LeakyReLU()(x)

x = layers.Reshape((200, 1))(x)

generator = keras.models.Model(generatorInput, x)

generator.summary()

###################
## discriminator ##
###################

discriminatorInput = layers.Input(shape = (height, width))

x = layers.Flatten()(discriminatorInput)

x = layers.Dense(128, activation = 'sigmoid')(x)
x = layers.Dense(64, activation = 'sigmoid')(x)
x = layers.Dense(32, activation = 'sigmoid')(x)
x = layers.Dense(1, activation = 'sigmoid')(x)

discriminator = keras.models.Model(discriminatorInput, x)

discriminatorOptimizer = keras.optimizers.RMSprop(lr = 0.001)

discriminator.compile(optimizer = discriminatorOptimizer, 
                      loss = 'binary_crossentropy')

discriminator.summary()

#########
## gan ##
#########

discriminator.trainable = False  # so that discriminator isn't updated during gan training

ganInput = keras.Input(shape = (latentDim, ))
ganOutput = discriminator(generator(ganInput))
gan = keras.models.Model(ganInput, ganOutput)

ganOptimizer = keras.optimizers.RMSprop(lr = 0.004)

gan.compile(optimizer = ganOptimizer, loss = 'binary_crossentropy')

####################
## generate meter ##
####################

def gen():
    gan.load_weights('Generate_Meter/models/favorable_models/gan11440.h5')  # adjust model weights loaded to taste

    randLatentVects = np.random.normal(0, 1, size = (1, latentDim))  

    generatedSong = generator.predict(randLatentVects)

    song = generatedSong[0] * 108
        
    np.savetxt('Generate_Meter/nnout/meter_generator_output/output.csv',
               song.reshape(200, 1),
               delimiter = ',')
           
