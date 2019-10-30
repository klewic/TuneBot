import mido
import os
import numpy as np
import pandas as pd  # pandas is inefficient for processing this data, but I've only got to run this script once so I'm not going to lose any sleep over it


def zeroAppend(track, end):
        if len(track) < end:
            track = np.vstack([track,
                               np.array([ [0,0,0]
                                          for _ in range(end - len(track)) ])
                               ])
        return track

    
outputFilename = 0  # data output will be given sequential numbers as filenames

for song in os.listdir(os.path.join('midis')):
    
    print('Importing song titled: ', song)
    
    try:
        songCurr = mido.MidiFile('midis/' + song)  # assign songCurr to mido interpretation of active/current midi file 
    except:
        print('song cannot be imported! :(')
        continue  # unless mido doesn't know what it is
    
    songNew = mido.MidiFile()
    t0 = mido.MidiTrack()
    songNew.tracks.append(t0)
    t0.append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(120)))
    t0.append(mido.MetaMessage('time_signature', numerator = 4, 
                                                 denominator = 4))  # prepare an empty midi file, songNew, to hold processed songCurr
    
    trackLen = []  # begin to process songNew such that its structure is exactly 5 tracks and track 5 is percussion
    trackChan = []                                                             
    
    if len(songCurr.tracks) < 5:  # because uniformity is key...
        print('Insufficient tracks in %r - cannot process - sorry!' % song)  # ...we reject any songs that have less than 5 tracks
        continue
        
    for i, track in enumerate(songCurr.tracks):
        trackLen.append(len(track))  # grab lengths of each track (note count)
        for msg in track:
            if msg.is_meta:  # skip tracks that don't hold note data
                pass
            elif hasattr(msg, 'channel'):  # this is how I detect real music tracks
                trackChan.append(  (msg.channel, i)  )  # (channel, track index #)
                break  # channel number recorded - get out of msg loop!
    
    print(trackLen)
    print(trackChan)
    
    if len(trackChan) > 10:
        print('Too many channels used! Resticting to 5 may sound bad')
        continue  # skip songs with many many channels - heuristic Im using to get better song snippets
    
    drumTrack = 'assign the track number of the channel 9 track to this var'
    for track in trackChan:
        if track[0] == 9:  # if the track's channel == 9, it's a drum track (midi file convention)
            drumTrack = track[1]  # so name the drum track's index number
            break   # then move on
            
    if isinstance(drumTrack, int):  # drumTrack is int if captured, string otherwise
        print('The drums exist for %r!\n' % song)
    else:
        print('The drums do not exist for %r :(\n' % song)                                             
        continue  # ignore a song without a drum track
        
    top5 = np.argpartition(trackLen, -5)[-5:]   # grab the index numbers of the top 5 tracks with the most notes
    if len(np.setxor1d(top5, drumTrack)) < 5:
        top4 = np.setxor1d(top5, drumTrack)
    else:
        top4 = np.argpartition(trackLen, -4)[-4:]  # to be used if the drum track is not in top5 (rare)
    
    for i, track in enumerate(songCurr.tracks):
        if i in top4:
            songNew.tracks.append(track)  # populate songNew with msgs from the top 4 length tracks + 5th drum track
    songNew.tracks.append(songCurr.tracks[drumTrack])  # at end to ensure track 5
    
    ##########################################################################
    ##### process songNew so that it can be interpretted by an algorithm #####
    ##########################################################################
    
    td = [np.array([], dtype = int).reshape(0,3) for _ in range(5)]  # td for track data
    startingTick = []  # hold the tick number right before each tracks initial note
    initial0s = []  # hold the amount of initial 0's at the start of each track - used later to cut out empty space
    string2int = {'note_on':1, 'note_off':0}
    
    for i in range(1, 6):  # note: track 0 is all meta data so we start at 1
        ix = i-1   # assign index number as (i - 1) because in td, td[0] is track 1
        timeAdd = 0  # for holding times of non note messages to add to note time attributes                                                               
        initNote = True
        for j, msg in enumerate(songNew.tracks[i]):
            
            if initNote:
                timeAdd += msg.time  # note that non note messages can have time attributes and time attribues are # of ticks since the last message
                
                if msg.type == 'note_on':   # when you find the first note...
                    td[ix] = np.vstack([td[ix],   # ...start stacking data!
                                        np.array([1, msg.bytes()[1], timeAdd],  # timeAdd will represent the amount of blank ticks before the instrument enters 
                                                  dtype = int)     
                                        ])
                    initNote = False  # now that weve found the initNote, we can forget about it for the rest of the loop
                    initial0s.append(timeAdd)  # let's add the number of initial 0's to the list before we...
                    timeAdd = 0   # ...reset timeAdd to 0
            elif msg.is_meta or msg.type not in {'note_on', 'note_off'}:
                timeAdd += msg.time  # if message is not a note, we don't care about anything except its time attribute...
            else:
                td[ix] = np.vstack([td[ix],
                                    np.array([string2int[msg.type],
                                              msg.bytes()[1],
                                              msg.time + timeAdd],  # ...because we add the times of all previous non note messages to the current note
                                             dtype = int)                 
                                    ])                                         
                timeAdd = 0
    
    print('td made on to td2')
    if min([len(x) for x in td]) == 0:
        print('td contains a length 0 track - skipping')
        continue  # td may contain a length 0 track if the drums are not filled in with more than meta messages
    
    td2 = [np.array([], dtype = int).reshape(0,3) for _ in range(5)]  # now lets map out the songs by tick          
    
    for i in range(5):             
        
        inc = 0
        notes = [0,0,0]  # at start, initialize all 3 notes as off
        
        td2[i] = np.vstack([td2[i],
                                np.array([notes 
                                          for _ in range(td[i][0][2] + 1)],  # +1 added to range because row 0 times of 0 were breaking my script
                                         dtype = int)
                            ])  # stack all note off rows until the start of the first note
        
        for j in range(len(td[i])):
            if inc > 0:   # if we incremented through the messages because of time 0 notes (used in chords for example)...
                inc -= 1
                continue  # ...then we continue the loop to bypass these notes (they've already been recorded)
            
            notesOn = set()  # prepare sets to hold the notes that are turned on...
            notesOff = set()  # ...or turned off at this time
            
            if td[i][j][0] == 1:
                notesOn.add(td[i][j][1])
            elif td[i][j][0] == 0:
                notesOff.add(td[i][j][1])  # we add the note of this iteration to notesOn or notesOff
    
            while(  (j + inc + 1) < len(td[i]) 
                    and
                    td[i][j + inc + 1][2] == 0  ):
                inc += 1
                if td[i][j + inc][0] == 1:
                    notesOn.add(td[i][j + inc][1])
                elif td[i][j + inc][0] == 0:
                    notesOff.add(td[i][j + inc][1])   # and also add any notes occuring at time 0 relative to the initial note
                
            if (j + inc + 1) < len(td[i]):  # if we didnt make it to the end of the track
                numTicks = td[i][j + inc + 1][2]  # we prepare to stack the notes for as many ticks as necessary
            else:  # but if we did make it to the end of the track
                td2[i] = np.vstack([td2[i],
                                np.array([0,0,0], dtype = int)
                                ])  # we stack a single set of notes, all 0, to turn any active notes off
                break  # and then we're done with this track
            
            for note in notesOff: 
                if note in notes:
                    notes[notes.index(note)] = 0  # if any notes in notes are in notesOff, we turn them off
                
            for note in notesOn:
                sumNotesOn = sum(note > 0 for note in notes)  # get the number of notes currently on
                if note in notes:
                    continue  # if a noteOn note is already on, we skip it
                elif sumNotesOn < 3:
                    notes[notes.index(0)] = note  # if theres room for a new note, we add it
                else:
                    break  # otherwise notes is full and we move on
                
            td2[i] = np.vstack([td2[i],
                                np.array([notes 
                                          for _ in range(numTicks)],
                                         dtype = int)
                                ])  # stack all relevant notes for this iteration
    
    start = min(initial0s)  # get the first tick thats not all silence (all notes off for all tracks)
    end = max([len(track) for track in td2])  # get the max length of all tracks
    
    
    
    
    t1 = zeroAppend(td2[0], end)  # add 0 rows of notes until all tracks same length
    t1 = t1.tolist()[start:]  # and start each track at a point where not all tracks are silent
    t2 = zeroAppend(td2[1], end)
    t2 = t2.tolist()[start:]
    t3 = zeroAppend(td2[2], end)
    t3 = t3.tolist()[start:]
    t4 = zeroAppend(td2[3], end)
    t4 = t4.tolist()[start:]
    t5 = zeroAppend(td2[4], end)
    t5 = t5.tolist()[start:]
    
    combinedTracks = []
    
    incrementor = 1
    for i in range(len(t1)):
        combinedTracks.append([t1[i][0],
                               t1[i][1],
                               t1[i][2],
                               t2[i][0],
                               t2[i][1],
                               t2[i][2],
                               t3[i][0],
                               t3[i][1],
                               t3[i][2],
                               t4[i][0],
                               t4[i][1],
                               t4[i][2],
                               t5[i][0],
                               t5[i][1],
                               t5[i][2]
                               ])
        #print(combinedTracks[i])
        if i == 0:
            prevRow = combinedTracks[i].copy()
            combinedTracks[i].append(incrementor)
        elif combinedTracks[i] == prevRow:
            combinedTracks[i].append(incrementor)
        else:
            incrementor += 1
            prevRow = combinedTracks[i].copy()
            combinedTracks[i].append(incrementor)
        #print(combinedTracks[i])
        #print(prevRow)
                   
    #print(combinedTracks)
    
    output = pd.DataFrame({'t11':list(zip(*combinedTracks))[0],
                           't12':list(zip(*combinedTracks))[1],
                           't13':list(zip(*combinedTracks))[2],
                           't21':list(zip(*combinedTracks))[3],
                           't22':list(zip(*combinedTracks))[4],
                           't23':list(zip(*combinedTracks))[5],
                           't31':list(zip(*combinedTracks))[6],
                           't32':list(zip(*combinedTracks))[7],
                           't33':list(zip(*combinedTracks))[8],
                           't41':list(zip(*combinedTracks))[9],
                           't42':list(zip(*combinedTracks))[10],
                           't43':list(zip(*combinedTracks))[11],
                           't51':list(zip(*combinedTracks))[12],
                           't52':list(zip(*combinedTracks))[13],
                           't53':list(zip(*combinedTracks))[14],
                           'rid':list(zip(*combinedTracks))[15]
                           })  # combine tracks into single df
    
    print(output.head())
    groups = output.groupby(['rid', 't11', 't12', 't13', 't21', 't22', 't23', 
                                    't31', 't32', 't33', 't41', 't42', 't43', 
                                    't51', 't52',  't53'])['rid'].count()
    
    rid, t11, t12, t13, \
         t21, t22, t23, \
         t31, t32, t33, \
         t41, t42, t43, \
         t51, t52, t53 = zip(*groups.index)
         
    noTicks = groups.values
    
    outputCondensed = pd.DataFrame({'t11':t11,
                                    't12':t12,
                                    't13':t13,
                                    't21':t21,
                                    't22':t22,
                                    't23':t23,
                                    't31':t31,
                                    't32':t32,
                                    't33':t33,
                                    't41':t41,
                                    't42':t42,
                                    't43':t43,
                                    't51':t51,
                                    't52':t52,
                                    't53':t53,
                                    'noTicks':noTicks
                                    })
    
    #print(outputCondensed)
    
    slices = len(outputCondensed) // 200
    
    for s in range(slices):
        if ( max(outputCondensed.loc[199*s:199*(s+1), 'noTicks']) > 420 ):
            print('Slice with >420 tick block omitted')
        else:
            outputFilename += 1
            outputFilenameWhole = str(outputFilename) + '.csv'
            outputCondensed.loc[199*s:199*(s+1), ].to_csv(
                    'dataOld/' + outputFilenameWhole, index = False)
            print(outputFilenameWhole)
            
