#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:00:44 2019

@author: aviallon
"""

import numpy as np
#import matplotlib.pyplot as plt
import evolvenn as enn

evolve = enn.Evolution(
        model=enn.Sequential(input_size=1, output_size=1,  layers=[
                enn.Dense(9, activation=enn.sigmoid, use_bias=False),
                enn.Dense(9, use_bias=False),
                enn.Dense(9, activation=enn.tanh, use_bias=False),
                enn.Dense(9, use_bias=False),
                enn.Dense(9, activation=enn.tanh, use_bias=False),
                enn.Dense(9, use_bias=False),
                enn.Dense(5, activation=enn.relu, use_bias=False),
                #enn.Dense(activation=enn.relu, output_size=24, lr=0.05, lr2=0.005),
                #enn.Dense(activation=enn.relu, output_size=24, lr=0.05, lr2=0.005),
                #enn.Dense(8),
                #enn.Dense(activation=enn.hard_tanh, output_size=8, lr=0.05),
                #enn.Dense(128),
                #enn.Dense(64, activation=enn.relu),
                #enn.Dense(8),
                enn.Dense(1, use_bias=False)]),
 bestN=6, numbers=70, compile=True)

x = np.linspace(0, 5**2, num=500)
y = np.sqrt(x)

evolve.evolve(batch_size=50, data=[x, y], epochs=100, generator=enn.Evolution.genetic, loss=enn.mean_squared_error, shuffle=False, decay=1e-3)

#x_test = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
x_test = [0, 1, 4, 9, 16, 25, 36]

y_pred = evolve.predict_best(x_test)
print(y_pred)