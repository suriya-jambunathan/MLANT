#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:58:45 2021

@author: suriyaprakashjambunathan
"""

import numpy as np

freq = 3.5 * (10**9)
c = 299792458
wavelength = c/freq

λ = wavelength * (10**6)
εr = 4.3
h = 200 #thickness

Wm = []
num = 0.025*λ 
while num <= (λ/4):
    Wm.append(num)
    num = num + (0.00625*λ)
    
W0m = []
for i in range(19,77):
    W0m.append(λ*i/10000)

dm = []
for i in range(9,57):
    dm.append(λ*i/10000)
    
#tm[i] = 0.1 * Wm[i]

tm = []
for i in Wm:
    tm.append(0.1 * i)
    
rows = []
for i in range(3,13):
    rows.append(i)
    
W = (c/(2*freq))*(np.sqrt(2/(εr + 1)))*(10**6)

εreff = ((εr + 1)/2) + ((εr - 1)/2)*1/((np.sqrt(1 + (12*h/W))))

del_L = 0.412*h*((εreff + 0.3)*((W/h) + 0.264))/((εreff - 0.258)*((W/h) + 0.8))

L = ((c/(2*freq*(np.sqrt(εreff))))*(10**6)) - (2*del_L)


tm = []
for i in Wm:
    tm.append(0.1 * i)
    
wm = Wm[0]

Xa_min = 0
Xa_max = (W/2) - (wm/2)
Ya_min = wm
Ya_max = (4*L/(min(rows) - 1)) - wm

Xa = []
num = Xa_min
while num <= Xa_max:
    Xa.append(num)
    num = num + 50

Ya = []
num = Ya_min
while num <= Ya_max:
    Ya.append(num)
    num = num + 50
    
Rows = rows
    
    


