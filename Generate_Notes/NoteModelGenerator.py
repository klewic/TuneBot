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

###################
## load the data ##
###################

def convertToOneHot(inputArr, indices):
    inputArrOneSection = inputArr[ indices ]
    oneHotArray = np.zeros(60)
    for noteNum in inputArrOneSection:
        if (noteNum < 28 or noteNum >= 88):
            pass
        else:
            oneHotArray[noteNum-28] = 1
    return oneHotArray

####################
## start training ##
####################

iterations = 40000
batchSize = 2#8
saveDir = 'nnout'
lossLog = []

start = 0

for step in range(iterations):
    
    print('on step ', step)
    
    xData = []
    files = [random.randint(1, 14208) for _ in range(batchSize)]  # I generated 14208 training files - adjust this to fit your training set size
    #print('Loading files: ', files)

    for filename in files:
        data = np.genfromtxt('../Generate_Data/data/' + str(filename) + '.csv', 
                             skip_header = 1, dtype = int, delimiter = ',')
        xDataEntry = []
        for row in data:
            ohnotes = convertToOneHot(row, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            xDataEntry.append( ohnotes.tolist() )
        xData.append(xDataEntry)
    
    # get generator to generate "fake" songs (note: the generator is not being trained in this part!)
    randLatentVects = np.random.normal(0, 1, size = (batchSize, latentDim))  
    
    generatedSongs = generator.predict(randLatentVects)
    
    print(generatedSongs[0][0])
    
    # pull a batch of "real" songs from xData
    realSongs = np.reshape(xData, (batchSize, 200, 60, 1))
    
    combinedSongs = np.concatenate([generatedSongs, realSongs])
    
    # the labels suggest that "fake" songs should be identified with a "1" i.e. "1" is a fakeness flag
    labels = np.concatenate([ np.ones(  (batchSize, 1) ),
                              np.zeros( (batchSize, 1) ) ] )
    labels += 0.05 * np.random.random(labels.shape)  # add noise to the labels - heuristic to improve gan performance
     
    dLoss = discriminator.train_on_batch(combinedSongs, labels) 
    
    randLatentVects = np.random.normal(size = (batchSize, latentDim)) 
    
    misleadingTargets = np.zeros((batchSize, 1))  # these are not the true labels for the targets - remember the generator needs to fool the discriminator or else it "loses" (hence loss)
    
    aLoss = gan.train_on_batch(randLatentVects, misleadingTargets)
    i = 0
    while ( (aLoss - dLoss > 4) and (i < 32) ):
        randLatentVects = np.random.normal(size = (batchSize, latentDim))
        aLoss = gan.train_on_batch(randLatentVects, misleadingTargets)
        i += 1
    
    start += batchSize
    if start > (len(xData) - batchSize):
        start = 0
        
    if step % 10 == 0:
        
        if aLoss < 2:
            gan.save_weights('models/gan' + str(step) + '.h5')
        
        print('discriminator loss: ', dLoss)
        print('adversarial loss: ', aLoss)
        
        dPreds = discriminator.predict_on_batch(combinedSongs)  # winner if first half 1's (predicts fake are fake) / second half 0's (predicts real are real)
        aPreds = gan.predict_on_batch(randLatentVects)  # winner if all 0's (fools discriminator into predicting all fake songs as real)
        
        print('discriminator preds: ', dPreds)
        print('adversarial preds: ', aPreds)
        
        lossLog.append(['step:', step, 
                        'dLoss:', dLoss, 
                        'aLoss:', aLoss,
                        'dPreds:', str(dPreds),
                        'aPreds:', str(aPreds)])
        
        song = generatedSongs[0]
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
        
        np.savetxt(os.path.join(saveDir, 'gensong' + str(step) + '.csv'),
                   processedSong.reshape(200, 15),
                   delimiter = ',')

        with open('lossLog.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lossLog)
