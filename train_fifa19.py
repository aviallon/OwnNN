#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:51:14 2019

@author: aviallon
"""

import numpy as np
import matplotlib.pyplot as plt
import evolvenn as enn
from evolvenn import Dense, relu, hard_sigmoid, Sequential, LeakyReLU,\
                    Adam, NoOptimizer, linear, hard_tanh, sigmoid
import pandas

data = pandas.read_csv('datasets/fifa19.csv')
data.dropna()
column_names = {'Name':'nom',
                'Age':'age',
                'Club':'club',
                'Overall':'score',
                'Potential':'potential',
                'Value':'price',
                'Wage':'wage',
                'Special':'special',
                'Skill Moves':'skill',
                'Acceleration':'accel',
                'SprintSpeed':'speed',
                'Crossing':'crossing',
                'Agility':'agility',
                'Reactions':'reactions',
                'Balance':'balance',
                'LongShots':'longshots',
                'Strength':'strength',
                'ShotPower':'power',
                'Jumping':'jump',
                'Stamina':'stamina',
                'Aggression':'agress',
                'International Reputation':'fame',
                'Work Rate':'work_rate',
                'Body Type':'body',
                'Height':'height',
                'Weight':'weight',
                'Finishing':'finish',
                'FKAccuracy':'fk_acc',
                'HeadingAccuracy':'head_acc',
                'ShortPassing':'short_pass',
                'Volleys':'volleys',
                'Dribbling':'dribbling',
                'Curve':'curve',
                'Vision':'vision',
                'Penalties':'penalties',
                'Composure':'composure',
                'Marking':'marking',
                'Interceptions':'intercept',
                'Positioning':'pos',
                'StandingTackle':'gk_standing_tackle',
                'SlidingTackle':'gk_slide_tackle',
                'GKDiving':'gk_diving',
                'GKHandling':'gk_handling',
                'GKKicking':'gk_kick',
                'GKPositioning':'gk_pos',
                'GKReflexes':'reflexes',
                'Release Clause':'release_clause'}

data = data.rename(columns=column_names)
data['release_clause'].replace(to_replace='[^0-9]+', value='',inplace=True,regex=True)
data['price'].replace(to_replace='[^0-9]+', value='',inplace=True,regex=True)
data['weight'].replace(to_replace='[^0-9]+', value='',inplace=True,regex=True)
data['wage'].replace(to_replace='[^0-9]+', value='',inplace=True,regex=True)
data['release_clause'] = data['release_clause'].astype(float)
data['price'] = data['price'].astype(float)
data['weight'] = data['weight'].astype(float)
data['wage'] = data['wage'].astype(float)
data = data.replace({"work_rate":{"High/ High":5, "High/ Medium":4, "Medium/ High":3, "Medium/ Medium":2, "Medium/ Low":1, "Low/ Medium":0, "Low/ Low":-1}})
data = data.replace({'body':{'Lean':0, 'Normal':1, 'Stocky':2, 'Messi':1, 'Neymar':2}})
is_numeric =  data.applymap(lambda x: isinstance(x, (int, float))).all(0)
to_examine = list(filter(lambda key:is_numeric.to_dict()[key] == False, is_numeric.to_dict().keys()))
ok = list(filter(lambda key:is_numeric.to_dict()[key] == True and key != 'score' and key != 'potential' and all(c.islower() for c in key), is_numeric.to_dict().keys()))
data = data.sample(frac=1).reset_index(drop=True)

data.dropna(subset=ok, how='any', axis=0, inplace=True)


X = data[ok]
Y = data[['score']]
X = (X - X.mean()) / (X.max() - X.min())
Y = (Y - Y.mean()) / (Y.max() - Y.min())

try:
    del(model)
except Exception:
    pass

model = Sequential(input_size=len(ok), output_size=1, layers=[
        Dense(80, activation=LeakyReLU(), dropout=0.5),
        Dense(32, activation=LeakyReLU(), dropout=0.25),
        Dense(16, activation=LeakyReLU(), dropout=0.25),
        Dense(1, activation=linear())
        ])

model.train(X.get_values()[:3000], Y.get_values()[:3000], batch_size=10, optimizer=Adam(epsilon=0.00001, lr=0.004), epochs=300, patience=15, lr_decay=1e-3)