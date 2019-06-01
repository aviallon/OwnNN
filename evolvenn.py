#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:02:02 2019

@author: aviallon
"""

import copy
import time
import numpy as np
from multiprocessing import Pool, Process, cpu_count

np.seterr(all='ignore')

floatx = np.float32

def linear(x):
    return x

def relu(x):
    return np.maximum(0.0, x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def hard_sigmoid(x):
    return np.clip((x*0.2)+0.5, 0, 1) 

def leaky_relu(x):
    if x >= 0:
        return x
    
    return 0.001*x

def tanh(x):
    return np.tanh(x)

def softmax(x):
    xnew = x
    try:
        xnew = np.exp(x)/np.sum(np.exp(x))
    except Exception as w:
        print("Warning :", w)
        
    return xnew

def LeakyReLU(alpha=1e-3):
    def func(x):
        if x >= 0:
            return x
        else:
            return alpha*x
        
    return func

def hard_tanh(x):
    return np.clip(x, -1.0, 1.0)

class Layer:
    id_counter = 0
    def __init__(self, output_size:int, activation=linear, lr = 1e-2, lr2 = 1e-3, trainable = True, use_bias=False):
        self.output_size = output_size
        self.activation = activation
        self.lr = lr
        self.lr2 = lr2
        self.trainable = trainable
        self.use_bias = use_bias
        self.uid = Layer.id_counter
        Layer.id_counter += 1
        self.M = None
        self.bias = None
        self.input_size = None
        self.name = "Layer"
    
    def out(self, x:np.array) -> np.array:
        pass
    
    def newid(self):
        self.uid = Layer.id_counter
        Layer.id_counter += 1
    
    def init(self, input_size:int):
        if self.trainable:
            self.M = np.random.normal(0, 1, (input_size, self.output_size)).astype(floatx)
            self.bias = np.random.normal(0, 1, self.output_size).astype(floatx)
        self.input_size = input_size
        
    def __repr__(self):
        return "{}(input_size = {}, output_size = {}, activation={}, trainable={})".format(self.name, self.input_size, self.output_size, self.activation.__name__, self.trainable)
    
    def get_weights(self):
        return self.M

    def get_bias(self):
        return self.bias
    
    def set_weights(self, M, bias=None):
        if M.shape == self.M.shape:
            self.M = M
            if bias is None:
                self.bias = bias
        else:
            raise ValueError("New weights do not have the same shape !!! ({} versus {})".format(M.shape, self.M.shape))
    
class Dense(Layer):
    def __init__(self, output_size:int, activation=linear, lr = 1e-2, lr2 = 1e-3, trainable = True, use_bias = True):
        Layer.__init__(self, output_size, activation, lr, lr2, trainable)
        self.name = "Dense"
    
    def out(self, x:np.array) -> np.array:
        if self.use_bias:
            return self.activation(np.dot(x, self.M) + self.bias)
        else:
            return self.activation(np.dot(x, self.M))
        
class Flatten(Layer):
    def __init__(self, output_size:int):
        Layer.__init__(self, output_size, trainable=False)
        self.name = "Flatten"
        
    def out(self, x:np.array) -> np.array:
        xnew = x.flatten()
        return xnew.reshape((1, xnew.shape[0]))
    
    def init(self, input_size:int):
        pass
    
class Sequential:
    id_counter = 0
    def __init__(self, input_size:int, output_size:int, layers=[]):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.uid = Sequential.id_counter
        Sequential.id_counter += 1
        self.loss = np.inf
        self.debug = False
        
    def newid(self):
        self.uid = Sequential.id_counter
        Sequential.id_counter += 1
        
    def add(self, layer:Layer):
        self.layers.append(layer)
        
    def Add(self, layers:[Layer]):
        for layer in layers:
            self.add(layer)
        
    def compile(self):
        if self.layers[-1].output_size != self.output_size:
            print("Warning : output size of last layer must be the same as specified in the model initialisation. Modifiying the last layer output shape according to that.")
            self.layers[-1].output_size = self.output_size
        
        for i in range(len(self.layers)):
            isize = self.input_size
            if i > 0:
                isize = self.layers[i-1].output_size
            
            self.layers[i].init(isize)
            
    def summary(self):
        print(self)
        
    def __repr__(self):
        m = "Sequential model (id={}, loss={}) with {} layers :\n".format(self.uid, self.loss, len(self.layers))
        for l in self.layers:
            m += str(l) + "\n"
        m += "========\n"
        return m
        
    def predict(self, x:np.array) -> np.array:
        if self.debug:
            print("Predict input ",x.shape)
        for l in self.layers:
            try:
                x = l.out(x)
            except ValueError as e:
                print(e, " in layer :",l)
                raise(e)
        return x.flatten()
    
    def predict_batch(self, x:[np.array], pool=None, debug=False) -> [np.array]:
        self.debug = debug
        if pool is None:
            pool = Pool(4)
        
        y = pool.map(self.predict, x)
        return np.array(y)
    
class Evolution:
    def __init__(self, model:Sequential, numbers=50, bestN=8, compile=True):
        self.models = []
        for i in range(numbers):
            new_model = model.__new__(Sequential)
            new_model.__dict__ = copy.deepcopy(model.__dict__)
            new_model.newid()
            if compile:
                new_model.compile()
                
            self.models.append(new_model)
                
                
        self.numbers = numbers
        self.bestN = bestN
        self.history = []
        self.lr_qty = 1.0
    
    def evolve_batch(self, loss:callable, generator:callable, x:[np.array], y:[np.array], lr_qty = 1e-3, pool = None):
        
        if pool is None:
            pool = Pool(4)
            
        for i, model in enumerate(self.models):
            y_pred = model.predict_batch(x, pool, debug=False)
            self.models[i].loss = loss([y, y_pred])
    
        self.models, best_loss = generator(self.models, self.numbers, self.bestN, lr_qty, pool = pool)
        
        #print(best_loss)
        
        return best_loss
        
    def evolve(self, loss:callable, generator:callable, data=[], validation=[], epochs=10, batch_size=100, shuffle=True, decay=1):
        pool = Pool(int(cpu_count()*1.5))
        
        batch_number = 1
        if batch_size is None:
            batch_number = 1
            batch_size = len(data[0])
        else:
            batch_number = len(data[0])//batch_size
        
        for epoch in range(epochs):
            t0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
            print("Epoch {}/{}".format(epoch+1, epochs))
            epoch_loss = 0
            if shuffle and batch_size != None:
                rng_state = np.random.get_state()
                np.random.shuffle(data[0])
                np.random.set_state(rng_state)
                np.random.shuffle(data[1])
                
            for batch in range(batch_number):
                m = "\r Evolving {}/{} (lr : {}) -- loss : ".format(batch+1, batch_number, self.lr_qty)
                best_loss = self.evolve_batch(loss, generator, data[0][batch*batch_size:(batch+1)*batch_size], data[1][batch*batch_size:(batch+1)*batch_size], pool=pool, lr_qty=self.lr_qty)
                if best_loss <= 1e-13 or np.isnan(best_loss):
                    best_loss = 1e-13
                try:
                    m += "{}, (log : {})".format(best_loss, int(np.log10(best_loss)))
                except ValueError as e:
                    print(e, best_loss)
                print(m, end=" "*10)
                epoch_loss = best_loss
            
            duration = time.clock_gettime_ns(time.CLOCK_REALTIME)-t0
            print("\n\t --- {} ms/inference".format(duration/batch_number/batch_size/1e6))
            self.lr_qty = 1.0/(decay*epoch+1)
            self.history.append(epoch_loss)
        
    @staticmethod
    def _genetic(model, parent1, parent2, lr_qty):
            for i, layer in enumerate(model.layers):
                if not(layer.trainable):
                    continue
                
                M = layer.get_weights()
                bias = layer.get_bias()
                
                for (index, val) in np.ndenumerate(M):
                    if np.random.random() > 0.5:
                        M[index] = parent1.layers[i].get_weights()[index]
                    else:
                        M[index] = parent2.layers[i].get_weights()[index]
                    
                    if np.random.random() < layer.lr2:
                        M[index] = np.random.uniform(-1, 1)
                    elif np.random.random() < layer.lr:
                        M[index] += np.random.uniform(-lr_qty, lr_qty)
                        
                for (index, val) in np.ndenumerate(bias):
                    if np.random.random() > 0.5:
                        bias[index] = parent1.layers[i].get_bias()[index]
                    else:
                        bias[index] = parent2.layers[i].get_bias()[index]
                    
                    if np.random.random() < layer.lr2:
                        bias[index] = np.random.uniform(-1, 1)
                    elif np.random.random() < layer.lr:
                        bias[index] += np.random.uniform(-lr_qty, lr_qty)
                
                layer.set_weights(M, bias)
                
            return model
        
    @staticmethod
    def genetic(models:[Sequential], number:int, bestN:int, lr_qty:float, pool = Pool(4), mt = True):
        models = sorted(models, key=lambda model: model.loss)
        #print(models[0].loss, models[-1].loss)
        bests = models[:bestN]
        
        for model in models[bestN:]:
            np.random.shuffle(bests)
            if mt:
                model = pool.apply_async(Evolution._genetic, args=(model, bests[0], bests[1], lr_qty))
            else:
                model = Evolution._genetic(model, bests[0], bests[1], lr_qty)
        
        return models, bests[0].loss
    
    def predict_best(self, X:np.array) -> np.array:
        self.models = sorted(self.models, key=lambda model: model.loss)
        return np.array(self.models[0].predict_batch(X))

def mean_squared_error(Y:[np.array]) -> float:
    y_true, y_pred = Y[0], Y[1]
    return float(np.sum((y_true - y_pred)**2, axis=None)/np.size(y_true))

def categorical_xtropy(Y:[np.array]) -> float:
    y_true, y_pred = Y[0], Y[1]
    #print("XTROPY:", y_true.shape, y_pred.shape)
    return -np.sum(y_true*np.log(y_pred), axis=None)/y_true.shape[0]


def to_categorical(labels, n_vec=None):
    new_labels = []
    if n_vec is None:
        n_vec = np.maximum(labels)
    for l in labels:
        label = np.zeros(n_vec)
        label[l] = 1
        new_labels.append(label)
    return np.array(new_labels)
        

def PackedFunction(func:callable):
    def func(X):
        return func(X[0], X[1])
    
    return func

def gendata(func:callable, n=10000, r=5):
    x = np.random.uniform(-r, r, n)
    y = func(x)
    return [x, y]