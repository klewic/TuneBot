# hey
# for all you (ubuntu based) Linux users trying to play midi files:
# sudo apt-get install freepats timidity timidity-interfaces-extra
# timidity -iA (to activate - may be required before running script depending on your system)
# timidity -ig (for gui; for in terminal replace "-ig" w filename)
# src: https://sfxpt.wordpress.com/2015/02/02/how-to-play-midi-files-under-ubuntu-linux/
# you're also gonna be missing a module 'rtmidi' to get it goin'
# installing that can be a real pain
# here's what worked for me:
# you may need some dependencies to get through the install
# I also needed to install, for example, libasound2-dev, pkg-config, and libjack-dev
# out of rtmidi, python-rtmidi, and rtmidi-python, rtmidi-python worked for me
# but mido expects python-rtmidi by default so be sure to change the backend
# and be sure to edit .../site-packages/mido/backends/rtmidi_python 
# so that input_names and output_names are equal to
# set(rtmidi.MidiIn(b'in').ports)
# and
# set(rtmidi.MidiOut(b'out').ports)
# respectively or you will get an error message such as "TypeError: expected bytes, str found"
# ACTUALLY, there are two more instances of MidiIn and MidiOut in that file that will have to be replaced with the b'in'/b'out' counterparts
# if you're not using python 3, this bug may not apply to you
# I didn't have to make such an edit while writing this script in python 2.7
# src: https://github.com/superquadratic/rtmidi-python/issues/17

import mido
mido.set_backend('mido.backends.rtmidi_python')  # changed the backend because I could only get rtmidi-python to work - not python-rtmidi
#print(mido.get_input_names())
#print(mido.get_output_names())
port = mido.open_output(b'TiMidity:0')  # set the port to play midi through
import os
import sys
import pandas as pd
import NoteGenerator as NG
import MeterGenerator as MG

inst1 = 1
inst2 = 1
inst3 = 1
inst4 = 1

clipSource = input("Type \'gen\' to generate a song " + 
                   "or \'train\' to play song from the training data:\n")

if clipSource == 'gen':
    NG.gen()
    MG.gen()
    filenameNotes = 'Generate_Notes/nnout/note_generator_output/output.csv'
    filenameMeter = 'Generate_Meter/nnout/meter_generator_output/output.csv' 
elif clipSource == 'train':
    whichSong = input("Which song?  Enter a number:\n")
    filenameNotes = 'Generate_Data/data/' + whichSong + '.csv'
    filenameMeter = 'Generate_Data/data/' + whichSong + '.csv'   
else:
    sys.exit("Invalid input!")

if input("Would you like to change the instruments used from their defaults?  y/n:\n") == "y":
    inst1 = int(input("Integer value for instrument 1:\n"))
    inst2 = int(input("Integer value for instrument 2:\n"))
    inst3 = int(input("Integer value for instrument 3:\n"))
    inst4 = int(input("Integer value for instrument 4:\n"))
                                      
vel = 50  # velocity constant across all notes/tracks for simplicity

if clipSource == "gen":
    dfNotes = pd.read_csv(filenameNotes, names = ['t11', 't12', 't13',
                                                  't21', 't22', 't23',
                                                  't31', 't32', 't33',
                                                  't41', 't42', 't43',
                                                  't51', 't52', 't53']).round(0).astype(int)

    dfMeter = pd.read_csv(filenameMeter, names = ['noTicks']).round(0).astype(int)
    
elif clipSource == "train":
    dfNotes = pd.read_csv(filenameNotes, usecols = ['t11', 't12', 't13',
                                                    't21', 't22', 't23',
                                                    't31', 't32', 't33',
                                                    't41', 't42', 't43',
                                                    't51', 't52', 't53']).round(0).astype(int)
    
    dfMeter = pd.read_csv(filenameMeter, usecols = ['noTicks']).round(0).astype(int)

dfMeter[dfMeter < 0] = 0
    
df = pd.concat([dfNotes, dfMeter], axis = 1)

def getTrackData(txdf):
    indicator = 1
    txdf['change'] = 0
    for i in range(len(txdf)):
        if i == 0:
            txdf.loc[i, 'change'] = indicator
        elif txdf.iloc[i, 0:3].tolist() == txdf.iloc[i-1, 0:3].tolist():
            txdf.loc[i, 'change'] = indicator
        else:
            indicator += 1
            txdf.loc[i, 'change'] = indicator
    txdf['noTicks'] = df['noTicks']
    txdf.columns = ['uno', 'dos', 'tres', 'change', 'noTicks']
    return txdf

t1df = getTrackData(df.loc[:, ['t11', 't12', 't13']])
t2df = getTrackData(df.loc[:, ['t21', 't22', 't23']])
t3df = getTrackData(df.loc[:, ['t31', 't32', 't33']])
t4df = getTrackData(df.loc[:, ['t41', 't42', 't43']])
t5df = getTrackData(df.loc[:, ['t51', 't52', 't53']])

def tick2codeMsg(df):
    groups = df.groupby(['change', 'uno', 'dos', 'tres'])['noTicks'].sum()
    change, uno, dos, tres = zip(*groups.index)
    time = groups.values
    output = pd.DataFrame({'uno':uno, 'dos':dos, 'tres':tres, 'time':time}).astype(int)
    return output                                                      

t1df = tick2codeMsg(t1df)
t2df = tick2codeMsg(t2df)
t3df = tick2codeMsg(t3df)
t4df = tick2codeMsg(t4df)
t5df = tick2codeMsg(t5df)  # group all track data by change indicator and count number of ticks until change

output = mido.MidiFile()  # prepare MidiFile to be made from track data
t0 = mido.MidiTrack()

t1 = mido.MidiTrack()
t1.append(mido.Message('program_change', channel=1, program=inst1, time=0)) 
t2 = mido.MidiTrack()
t2.append(mido.Message('program_change', channel=2, program=inst2, time=0))
t3 = mido.MidiTrack()
t3.append(mido.Message('program_change', channel=3, program=inst3, time=0))
t4 = mido.MidiTrack()
t4.append(mido.Message('program_change', channel=4, program=inst4, time=0))
t5 = mido.MidiTrack()
t5.append(mido.Message('program_change', channel=9, program=1, time=0))  # set track instruments

output.tracks.append(t0)
output.tracks.append(t1)
output.tracks.append(t2)
output.tracks.append(t3)
output.tracks.append(t4)
output.tracks.append(t5)  # add tracks with info msgs to output

t0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(50), time = 0))
t0.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time = 0))  # set tempo and time signature of output

def trackWrite(data, track, chan, vel):  # write the note data to the song tracks
    for i in range(len(data)): 
        row = data.loc[i, ].tolist()  # row is the ith row of notes and time
        notes = row[0:3]  # separate out the notes part
        notesOn = sum(note > 0 for note in notes)  # count how many notes are on
        noteSet = set(notes)  # and convert to a set since it has useful properties for this purpose (and's and or's)
        try:
            noteSet.remove(0)  # remove the note off indicator as it is uninteresting and will only complicate things
        except KeyError:  # unless there are no 0 notes
            pass  # then do nothing
        
        if i == 0:  # first row
            time = row[3]  # set the time variable immediately since times in the row are offset from track msg times by 1 row
            prevNoteSet = noteSet  # these notes will be the previous notes
            continue  # because we will continue through the loop at this point if we are on the first iteration
        
        if notesOn == 0:  # if there are no notes on
            initNote = True
            for note in prevNoteSet:  # for every note that was on
                if initNote:
                    track.append(mido.Message('note_off', 
                                              channel = chan, 
                                              note = note, 
                                              velocity = vel,
                                              time = time))  # turn it off!
                    initNote = False
                else:
                    track.append(mido.Message('note_off', 
                                              channel = chan, 
                                              note = note, 
                                              velocity = vel,
                                              time = 0))  # in case you have to turn off multiple notes
            
        elif notesOn > 0:  # if at least one note is on
            initNote = True
            for note in (noteSet | prevNoteSet):  # for note amongst all current and previous notes
                if note in (noteSet & prevNoteSet):  # if the note is in both the previous notes and the current
                    continue  # then it hasn't been turned on or off
                elif (note in noteSet) and initNote:  # else if it's a current note (and not a prev note)
                    track.append(mido.Message('note_on', 
                                              channel = chan, 
                                              note = note, 
                                              velocity = vel,
                                              time = time))  # then we turn it on
                    initNote = False
                elif (note in noteSet):
                    track.append(mido.Message('note_on', 
                                              channel = chan, 
                                              note = note, 
                                              velocity = vel,
                                              time = 0))  # in case there's more than one to turn on
                elif (note in prevNoteSet) and initNote:  # else if it's a prev note (and not a current note)
                    track.append(mido.Message('note_off', 
                                              channel = chan, 
                                              note = note, 
                                              velocity = vel,
                                              time = time))  # then we turn it off
                    initNote = False
                elif (note in prevNoteSet):
                    track.append(mido.Message('note_off', 
                                              channel = chan, 
                                              note = note, 
                                              velocity = vel,
                                              time = 0))   # in case there's more than one to turn off
                    
        prevNoteSet = noteSet
        time = row[3]

trackWrite(t1df, t1, 1, vel)
trackWrite(t2df, t2, 2, vel)
trackWrite(t3df, t3, 3, vel)
trackWrite(t4df, t4, 4, vel)
trackWrite(t5df, t5, 9, vel)

for msg in output.play():
    print(msg)
    port.send(msg)  # play the song!

output.save('theSong.mid')
