import os
import csv
import numpy as np
import random
import keras
from keras import layers

latentDim = 16  # dimension of random noise data
height = 200
width = 60

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

x = layers.Reshape((200, 60, 1))(x)

generator = keras.models.Model(generatorInput, x)

generator.summary()

###################
## discriminator ##
###################

discriminatorInput = layers.Input(shape = (height, width, 1))

x = layers.Flatten()(discriminatorInput)

x = layers.Conv2D(64, 2, padding = 'same')(discriminatorInput)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 2, padding = 'same')(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 2, padding = 'same')(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)

x = layers.Dense(128)(x)
x = layers.LeakyReLU()(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(64, activation = 'sigmoid')(x)

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

#############
## predict ##
#############

def gen():
    saveDir = "nnout"

    gan.load_weights('Generate_Notes/models/favorable_models/gan1330.h5')  # adjust model weights loaded to taste

    randLatentVect = np.random.normal(0, 1, size = (1, latentDim))

    generatedSong = generator.predict(randLatentVect)

    song = generatedSong[0]
    processedSong = []
    for row in song:
        rowNotes = row[0:60].tolist()
        rowDrums = [ [0] for _ in range(60) ]
        processedRow = []
            
        top12 = sorted(rowNotes, reverse = True)[:12]
            
        for entry in top12:
            if entry[0] > 0.5:
                note = rowNotes.index(entry) + 28
                rowNotes[note - 28] = 0  # if probs are all equal to 1, this stops the same note from being recorded 3 times
            else:
                note = 0
            processedRow.append(note)

        top3 = sorted(rowDrums, reverse = True)[:3]

        for entry in top3:
            if entry[0] > 0.5:
                note = rowDrums.index(entry) + 28
                rowDrums[note - 28] = 0
            else:
                note = 0
            processedRow.append(note)
                
        processedSong.append(processedRow)
        
    processedSong = np.array(processedSong)
        
    np.savetxt('Generate_Notes/nnout/note_generator_output/output.csv',
            processedSong.reshape(200, 15),
            delimiter = ',')
