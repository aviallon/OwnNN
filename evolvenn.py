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
import os
import sys
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

np.seterr(all='ignore')

floatx = np.float32

class Activation:
    def __init__(self):
        self.__name__ = "Activation"
    
    def __call__(self, x):
        pass
    
    def deriv(self, x):
        pass
    
class linear(Activation):
    def __call__(self, x):
        return x
    
    def deriv(self, x):
        return 1
    
    def __init__(self):
        self.__name__ = "linear"

class relu(Activation):
    def __call__(self, x):
        return np.maximum(0.0, x)
    
    def deriv(self, x):
        def _kernel(x):
            if x <= 0:
                return 0
            else:
                return 1
            
        return np.vectorize(_kernel)(x)
    
    def __init__(self):
        self.__name__ = "relu"

class sigmoid(Activation):
    def __call__(self, x):
        return 1/(1+np.exp(-x))
    
    def deriv(self, x):
        sig_x = self.__call__(x)
        return sig_x*(1-sig_x)
    
    def __init__(self):
        self.__name__ = "sigmoid"

class hard_sigmoid(Activation):
    def __call__(self, x):
        return np.clip((x*0.2)+0.5, 0, 1)
    
    def deriv(self, x):
        def _kernel(x):
            if x > -2.5 and x < 2.5:
                return 1
            else:
                return 0
        
        return np.vectorize(_kernel)(x)
    
    def __init__(self):
        self.__name__ = "hard_sigmoid"

class tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    
    def deriv(self, x):
        return (1 - np.tanh(x)**2)
    
    def __init__(self):
        self.__name__ = "tanh"

class softmax(Activation):
    def __init__(self):
        self.__name__ = "softmax"
    
    def __call__(self, x):
        xnew = x.copy()
        try:
            xnew = np.exp(x)/np.sum(np.exp(x))
        except Exception as w:
            print("Warning :", w)
        return xnew
    
    def deriv(self, x):
        raise(NotImplementedError("NNNOOOOOO DON'T USE SOFTMAX YET !"))

class LeakyReLU(Activation):
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.__name__ = "LeakyReLU({})".format(alpha)
    
    def __call__(self, x):
        def _kernel(x):
            if x >= 0:
                return x
            else:
                return self.alpha*x
            
        return np.vectorize(_kernel)(x)
        
    def deriv(self, x):
        def _kernel(x):
            if x >= 0:
                return 1
            else:
                return -self.alpha
        
        return np.vectorize(_kernel)(x)
        

class hard_tanh(Activation):
    def __init__(self):
        self.__name__ = "hard_tanh"
    
    def __call__(self, x):
        return np.clip(x, -1.0, 1.0)
    
    def deriv(self, x):
        def _kernel(x):
            if x > -1 and x < 1:
                return 1
            else:
                return 0
        
        return np.vectorize(_kernel)(x)
    
class Loss:
    def __call__(self, y_true, y_pred) -> float:
        pass
    
    def deriv(self, y_true, y_pred) -> float:
        pass

class mean_squared_error(Loss):
    def __call__(self, y_true, y_pred) -> float:
        return float(np.sum((y_true - y_pred)**2, axis=None)/len(y_true))
    
    def deriv(self, y_true, y_pred) -> float:
        return y_true - y_pred

class categorical_xtropy(Loss):
    def __call__(self, y_true, y_pred):
        #print("XTROPY:", y_true.shape, y_pred.shape)
        return -np.sum(y_true*np.log(y_pred), axis=None)/y_true.shape[0]

class Layer:
    id_counter = 0
    def __init__(self, output_size:int, activation=linear, lr = 1e-3, lr2 = 1e-4, lr_bias = 1e-4, trainable = True, use_bias=False):
        self.output_size = output_size
        self.activation = activation
        self.lr = lr
        self.lr2 = lr2
        self.lr_bias = lr_bias
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
            self.bias = np.random.normal(0, 1)
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
            if not(bias is None):
                self.bias = bias
        else:
            raise ValueError("New weights do not have the same shape !!! ({} versus {})".format(M.shape, self.M.shape))
            
    def set_bias(self, bias):
        self.bias = bias
    
class Dense(Layer):
    def __init__(self, output_size:int, activation=linear, lr = 1e-2, lr2 = 1e-3, lr_bias=1e-6, bias_only = False, trainable = True, use_bias = True):
        Layer.__init__(self, output_size, activation, lr, lr2, lr_bias, trainable, use_bias)
        self.bias_only = bias_only
        self.name = "Dense"
    
    def out(self, x:np.array) -> np.array:
        if self.bias_only:
            return self.activation(np.ones((1, self.output_size))*self.bias)
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
    def __init__(self, input_size:int, output_size:int, layers=[], compile=True, debug=False):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.uid = Sequential.id_counter
        Sequential.id_counter += 1
        self.losses = [np.inf]
        self.val_loss = np.inf
        self.history = {}
        self.debug = debug
        self.clipnorm = 1.0
        if compile:
            self.compile()
        
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
    
    def predict_batch(self, x:[np.array], pool=Pool(4)) -> [np.array]:
        y = []
        if pool is None:
            if self.debug:
                print("No multithreading", file=sys.stderr)
            for el in x:
                y.append(self.predict(el))
        else:
            y = pool.map(self.predict, x)
        return np.array(y)
    
    
    def train_online(self, x:np.array, y:np.array, loss:Loss):
        # Forward propagation
        outputs = [x.copy()]
        for l in self.layers:
            try:
                x = l.out(x)
                outputs.append(x.copy())
            except ValueError as e:
                print(e, "in layer:", l)
                raise(e)
                
        self.losses.append(loss(y, outputs[-1]))
        if self.debug:
            #print("\t\t\t--------", self.loss)
            print(outputs)
                
        # Backward propagation
        
        # Calcul des erreurs
        deltas = [0]*len(self.layers)
        l = len(self.layers) - 1
        while l >= 0:
            errors = np.zeros(self.layers[l].M.shape[1]+1)
            for i in range(self.layers[l].M.shape[1]):
                delta = 0
                if l != len(self.layers) - 1:
                    if self.debug:
                        print(deltas[l+1])
                    for neurone in range(self.layers[l+1].M.shape[1]): # Nombre de sorties
                        delta += self.layers[l+1].M[i, neurone] * deltas[l+1][neurone]
                    
                    delta += self.layers[l+1].bias
                else:
                    delta = loss.deriv(y[i], outputs[l+1][i]) # On a n plus un outputs (ne pas oublier la couche d'entrée !)
                    
                delta *= self.layers[l].activation.deriv(outputs[l+1][i])
                errors[i] = delta
            
            # BIAS TRAINING:
            delta = 0
            if l != len(self.layers) - 1:
                for neurone, val in np.ndenumerate(self.layers[l+1].M):
                    delta += self.layers[l+1].M[neurone] * deltas[l+1][neurone[1]]
            else:
                for i in range(len(y)):
                    delta += loss.deriv(y[i], outputs[l+1][i])
            
            for i in range(len(outputs[l+1])):
                delta *= self.layers[l].activation.deriv(outputs[l+1][i])
            errors[-1] = delta
            
                
            deltas[l] = errors
            l -= 1
            
        # Entrainement
        for l in range(len(self.layers)):
            for poids, val in np.ndenumerate(self.layers[l].M):
                dErr_dweight = self.layers[l].lr * deltas[l][poids[1]]
                if l != len(self.layers) - 1:
                    dErr_dweight *= outputs[l][poids[0]]
                
                if abs(dErr_dweight) > self.clipnorm:
                    dErr_dweight = np.sign(dErr_dweight)*self.clipnorm
                self.layers[l].M[poids] += dErr_dweight
                
            dErr_dbias = self.layers[l].lr_bias * deltas[l][-1]
            if abs(dErr_dbias) > self.clipnorm:
                    dErr_dbias = np.sign(dErr_dbias)*self.clipnorm
                    
            self.layers[l].bias += dErr_dbias
        
    def train(self, X:[np.array], Y:[np.array], loss=Loss, epochs=100, validation_split=0.1):
        ntrain = int((1-validation_split)*len(X))
        Xtrain = X[:ntrain]
        Ytrain = Y[:ntrain]
        Xval = X[ntrain:]
        Yval = Y[ntrain:]
        self.history = {"loss":[np.inf], "val_loss":[np.inf], "epoch":[0]}
        for epoch in range(1, epochs+1):
            self.losses = []
            rng_state = np.random.get_state()
            np.random.shuffle(Xtrain)
            np.random.set_state(rng_state)
            np.random.shuffle(Ytrain)
            for i in range(len(Xtrain)):
                self.train_online(Xtrain[i].copy(), Ytrain[i].copy(), loss)
                if self.debug:
                    time.sleep(1)
            for i in range(len(Xtrain)-1,0,-1):
                self.train_online(Xtrain[i].copy(), Ytrain[i].copy(), loss)
            
            train_loss = np.mean(self.losses)
            
            Yval_pred = self.predict_batch(Xval, pool=None)
            self.val_loss = loss(Yval, Yval_pred)
            self.history['loss'].append(train_loss)
            self.history["val_loss"].append(self.val_loss)
            self.history['epoch'].append(epoch)
            print("Epoch {} - loss = {}, val_loss= {}".format(epoch, train_loss, self.val_loss))
        
    def plot_history(self):
        import matplotlib.pyplot as plt
        plt.close()
        plt.plot(self.history['epoch'], self.history['loss'], label='loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='val_loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.show()
    
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
    
    def evolve_batch(self, loss:Loss, generator:callable, x:[np.array], y:[np.array], lr_qty = 1e-3, pool = None):
        
        if pool is None:
            pool = Pool(4)
            
        for i, model in enumerate(self.models):
            y_pred = model.predict_batch(x, pool)
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
                
                if np.random.random() > 0.5:
                    bias = parent1.layers[i].get_bias()
                else:
                    bias = parent2.layers[i].get_bias()
                
                if np.random.random() < layer.lr2:
                    bias = np.random.uniform(-1, 1)
                elif np.random.random() < layer.lr:
                    bias += np.random.uniform(-lr_qty, lr_qty)
                
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