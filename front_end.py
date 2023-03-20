from matplotlib.ft2font import HORIZONTAL
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
import pandas as pd
import time
import streamlit as st
import io
import json
import math
import os

#code by Hans Heje



def display_fractal(fractal,placeholder,caption):
    plt.figure(figsize=(5, 5))
    plt.imshow(fractal, cmap='magma')
    plt.axis('off')
    #save_fractal_image(fractal)

    fig = plt.gcf()
    fig.set_size_inches(5, 5)  # Resize plot if needed
    buf = io.BytesIO()
    plt.savefig(buf, format='png',bbox_inches='tight')
    plt.close()
    buf.seek(0)
    placeholder.image(buf, use_column_width=True, caption=caption)

def save_fractal_image(fractal):
    list_fractal = fractal.tolist()
    json_fractal = json.dumps(list_fractal)
    recovered_fractal = json.loads(json_fractal)
    plt.figure(figsize=(5, 5))
    plt.imshow(recovered_fractal, cmap='magma')
    plt.axis('off')
    plt.savefig(f'images/{hash(str(fractal))}.png', bbox_inches='tight',pad_inches = 0)

def animate_transition(fractal1,fractal2,placeholder,caption):
    fractal1 = np.array(fractal1)
    fractal2 = np.array(fractal2)

    for i in range(20+1):
        display_fractal(fractal1*(1-i/20)+fractal2*(i/20),placeholder,caption)
        time.sleep(0.1)

@st.cache_data
def initialize():
    amount_of_fractals = 0
    with open("fractals.txt", "r") as fp:
        amount_of_fractals = len(json.load(fp))
    return os.path.getmtime(file_path),amount_of_fractals

header = st.container()
with header:
    st.title("Quantum visualization using fractals")

body = st.container()

col1, col2 = st.columns(2)


file_path = 'fractals.txt'

last_modified_time,amount_of_fractals = initialize()

 # Get the current modification time of the file
current_modified_time = os.path.getmtime(file_path)
with body:

    #radio button to select highlighted fractal
    fractal_id = st.radio(
    "Which fractals transition to highlight?",
    np.arange(1,amount_of_fractals+1),horizontal=True)

    #checkbox to select what other fractals to display underneeth
    # Create a horizontal layout for the checkboxes
    checkbox_labels = []
    st.write('Additional fractals to display')
    for i in range(amount_of_fractals):
        checkbox_labels.append(f'{i+1}')
    # Create a list to store the boolean values of each checkbox
    checkbox_values = [True] * len(checkbox_labels)

    # Create a horizontal layout for the checkboxes
    cols = st.columns(len(checkbox_labels))

    # Add a checkbox to each column
    for i, col in enumerate(cols):
        checkbox_values[i] = col.checkbox(checkbox_labels[i], value=checkbox_values[i])

    if st.button("reload fractals"):
        current_modified_time = os.path.getmtime(file_path)
        # Compare the current modification time with the previous one
        if current_modified_time != last_modified_time:
            placeholder = st.empty()
            fractals = []
            old_fractals = []
            with open("old_fractals.txt", "r") as fp:
                old_fractals = json.load(fp)
            with open("fractals.txt", "r") as fp:
                fractals = json.load(fp)
            placeholder.empty()
            animate_transition(old_fractals[fractal_id-1],fractals[fractal_id-1],placeholder,f'Fractal {fractal_id}')
            amount_of_fractals = len(fractals)
            old_fractals = fractals
            # Update the last modified time to the current one
            last_modified_time = current_modified_time
            # Add an image to the first column
            
            num_cols = sum(checkbox_values)

            if num_cols >0:
                # Create the columns
                cols = st.columns(num_cols)

                fractals_to_display = [i for i, x in enumerate(checkbox_values) if x]
                # Add an image to each column
                for i, col in enumerate(cols):
                    index = fractals_to_display[i]
                    # Print the list of checkbox values
                    display_fractal(fractals[index],col,f'Fractal {index+1}')




 








