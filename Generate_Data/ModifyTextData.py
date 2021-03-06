# used to restrict the range of acceptable notes in a song - heuristic to build cleaner dataset

import os
import pandas as pd

newFilenameInt = 1

for file in os.listdir('dataOld/'):
    df = pd.read_csv('dataOld/' + file)
    noteSet = set()
    noteSet.update(df.t11.tolist())
    noteSet.update(df.t12.tolist())
    noteSet.update(df.t13.tolist())
    noteSet.update(df.t21.tolist())
    noteSet.update(df.t22.tolist())
    noteSet.update(df.t23.tolist())
    noteSet.update(df.t31.tolist())
    noteSet.update(df.t32.tolist())
    noteSet.update(df.t33.tolist())
    noteSet.update(df.t41.tolist())
    noteSet.update(df.t42.tolist())
    noteSet.update(df.t43.tolist())
    noteSet.update(df.t51.tolist())
    noteSet.update(df.t52.tolist())
    noteSet.update(df.t53.tolist())
    
    noteSet.remove(0)
    
#    print(min(noteSet))
#    print(max(noteSet))
    if min(noteSet) >= 28 and max(noteSet) < 88:  # len 60 is 28..87 inclusive
        #print('all good')
        df.to_csv('data/' + str(newFilenameInt) + '.csv', index = False)
        newFilenameInt += 1
    else:
        #print('no good!')
        pass
    
