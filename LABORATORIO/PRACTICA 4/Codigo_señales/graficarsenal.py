# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:26:09 2023

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt

signal = np.loadtxt('./EkG1.txt');


plt.plot(signal)
plt.title('Señal EKG');plt.ylabel("A [uV]");plt.xlabel("#de muestras")

plt.savefig("SeñalEkG.jpg");

plt.show();