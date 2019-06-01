#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:00:44 2019

@author: aviallon
"""
import pyximport; pyximport.install()
import numpy as np
#import matplotlib.pyplot as plt
import evolvenn as enn
import keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32)[:100]
x_test = x_test.astype(np.float32)[:100]
x_train /= 255
x_test /= 255
y_train = enn.to_categorical(y_train[:100], 10)
y_test = enn.to_categorical(y_test[:100], 10)

evolve = enn.Evolution(
        model=enn.Sequential(input_size=28**2, output_size=10,  layers=[
                enn.Flatten(28**2),
                enn.Dense(128, activation=enn.hard_sigmoid, use_bias=False),
                enn.Dense(64, activation=enn.hard_sigmoid, use_bias=False),
                enn.Dense(32, activation=enn.hard_sigmoid, use_bias=False),
                enn.Dense(16, activation=enn.hard_sigmoid, use_bias=False),
                enn.Dense(10, activation=enn.softmax, use_bias=False)]),
 bestN=6, numbers=70, compile=True)

evolve.evolve(batch_size=4, data=[x_train, y_train], epochs=100, generator=enn.Evolution.genetic, loss=enn.categorical_xtropy, shuffle=True, decay=1e-3)

y_pred = evolve.predict_best(x_test[:100])
print(y_pred)
