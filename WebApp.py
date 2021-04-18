#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:42:41 2021

@author: suriyaprakashjambunathan
"""

#Description
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
from src.classes import *
from src.constants import Wm, W0m, dm, tm, Rows, Xa, Ya

import pyutilib.subprocess.GlobalData

'''
import signal
origsignal = signal.signal
signal.signal = lambda x, y: None
from socketio.asyncio_server import AsyncServer
signal.signal = origsignal
'''

import ptvsd
#ptvsd.enable_attach(address=('localhost', 8501))

#Title and sub-title
image = Image.open('Content/fyp_cst_ss.png')
#st.set_page_config(page_title='M.L.A.N.T.', page_icon = image, layout = 'wide', initial_sidebar_state = 'auto')
st.write("""
# Antenna Prediction
Find the dimensions for best performance
""")

#Open and display image on webapp

st.image(image, caption = 'GA Class_reg', use_column_width = True)

df = pd.read_csv('Content/antenna.csv')

st.subheader('Data Information: ')

#Show the data as a table
st.dataframe(df)

#Show statistics on data
st.write(df.describe())

#Show the data as a chart
chart = st.bar_chart(df)



best = [2141.3747, 650.78, 214.13747, 77.0894892, 7, 1200, 2141.3747]

#Get Feature inout from users
def get_user_input():
    wm = st.sidebar.slider('Wm (in µm)', min_value = Wm[0], max_value = Wm[-1], value = best[0], step = (Wm[1] - Wm[0]))
    w0m  = st.sidebar.slider('W0m (in µm)', min_value = W0m[0], max_value = W0m[-1], value = best[1], step = (W0m[1] - W0m[0]))
    d    = st.sidebar.slider('dm (in µm)', min_value = dm[0], max_value = dm[-1], value = best[2], step = (dm[1] - dm[0]))
    t    = st.sidebar.slider('tm (in µm)', min_value = tm[0], max_value = tm[-1], value = best[3], step = (tm[1] - tm[0]))
    rows = st.sidebar.slider('Rows', min_value = Rows[0], max_value = Rows[-1], value = best[4], step = (Rows[1] - Rows[0]))
    xa   = st.sidebar.slider('Xa (in µm)', min_value = Xa[0], max_value = Xa[-1], value = best[5], step = (Xa[1] - Xa[0]))
    ya   = st.sidebar.slider('Ya (in µm)', min_value = Ya[0], max_value = Ya[-1], value = best[6], step = (Ya[1] - Ya[0]))
    
    #store a dictionary into a variable
    user_data = {'Wm'   : wm,
                 'W0m'  : w0m,
                 'dm'   : dm,
                 'tm'   : tm,
                 'Rows' : rows,
                 'Xa'   : xa,
                 'Ya'   : ya}
    
    #transform the data into a dataframe
    #features = user_data.values
    #features = list(features)
    return([wm, w0m, d, t, rows, xa, ya])
    
#Store the user inputs
user_input = get_user_input()

#Set a subheader and display user input
st.subheader('User Input: ')
st.write(user_input)

'''
#Store model predictions in a variable
prediction = model.predict(user_input)

#Set a subheader and display prediction
st.subheader('Prediction: ')
st.write(prediction)
'''


#"/Users/suriyaprakashjambunathan/WebApp.py"
    


import sys
from streamlit import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "WebApp.py"]
    sys.exit(stcli.main())
    
    
