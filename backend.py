import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
import pandas as pd
import time
import io
import json
import math
import os
from src.Fractal import julia_set_jit
#code by Hans Heje



def load_statevectors():
    print("="*50)
    df = pd.read_csv('statevectors.csv')
    df.dropna(inplace=True)
    svid = df['svid'].tolist()
    statevectors = df['statevector'].tolist()

    #convert from string to list of complex numbers
    vfunc = np.vectorize(eval)
    for i in range(len(statevectors)):
        statevectors[i] = vfunc(statevectors[i]).tolist()
    return statevectors,svid

def create_all_fractals():
    statevectors,svid = load_statevectors()
    fractals = []
    #json_fractals = []
    max_iterations = 100
    size = 200

    id_cnt = 1
    for i in range(len(statevectors)):
        #if no statevector at that id is found, then create a blank fractal (not a fractal)
        while(id_cnt != svid[i]):
            fractals.append(np.full((size, size), max_iterations-1).tolist())
            id_cnt = id_cnt+1
        id_cnt = id_cnt+1

        #generate fractal
        number_of_qubits = int(math.log2(len(statevectors[i])))
        fractal = julia_set_jit(statevector_data=statevectors[i],number_of_qubits=number_of_qubits,max_iterations = max_iterations,size=size)
        fractals.append(fractal.tolist())
        #json_fractals.append(json.dumps(fractal.tolist()))
    
    return fractals

fractals = []
old_fractals = []
backup_fractals = []


# Set the path to the file you want to monitor
file_path = 'statevectors.csv'
# Get the initial modification time of the file
last_modified_time = os.path.getmtime(file_path)
while(True):
    # Get the current modification time of the file
    current_modified_time = os.path.getmtime(file_path)

    # Compare the current modification time with the previous one
    if current_modified_time != last_modified_time:
        print("="*50)
        print('The statevectors have been modified!')
        start_time = time.time()
        
        #backup last iteration of fractals
        with open("fractals.txt", "r") as fp:
            backup_fractals = json.load(fp)
        
        #get old fractals
        with open("old_fractals.txt", "r") as fp:
            old_fractals = json.load(fp)

        fractals = create_all_fractals()

        with open("fractals.txt", "w") as fp:
            json.dump(fractals, fp)

        #check if fractal has changed, if it hasnt changed, then keep the old
        for i in range(len(fractals)):
            if fractals[i] != backup_fractals[i]:
                old_fractals[i] = backup_fractals[i]
        
        with open("old_fractals.txt", "w") as fp:
            json.dump(old_fractals, fp)
        
        # Update the last modified time to the current one
        last_modified_time = current_modified_time
        print("--- The fractals has been updated, took: %s seconds ---" % (time.time() - start_time))
    time.sleep(1)




    





