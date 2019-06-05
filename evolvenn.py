#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:02:02 2019

@author: aviallon
"""
from __future__ import print_function, division
import copy
import time
import numpy as np
from multiprocessing import Pool, cpu_count
#import os
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
    
    
class Optimizer:
    def __init__(self):
        pass
    
    def init(self, layers):
        pass
    
    def step(self):
        self.bias += 1
        
    def optimize(self, l, neuron, weight, weight_grad):
        pass
    
    def optimize_bias(self, l, bias, bias_grad):
        pass
    
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, wd=1e-4):
        self.lr = lr
        self.lr_bias = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wd = wd
        self.moving_avg_grad = []
        self.moving_avg_grad_bias = []
        self.bias = 1
        self.moving_avg_squared_grad = []
        self.moving_avg_squared_grad_bias = []
    
    def init(self, layers):
        for l in range(len(layers)):
            if layers[l].trainable:
                self.moving_avg_grad.append(np.zeros(layers[l].getW().shape, dtype=floatx))
                self.moving_avg_squared_grad.append(np.zeros(layers[l].getW().shape, dtype=floatx))
                self.moving_avg_grad_bias.append(floatx(0))
                self.moving_avg_squared_grad_bias.append(floatx(0))
            else:
                self.moving_avg_grad.append(None)
                self.moving_avg_squared_grad.append(None)
                self.moving_avg_grad_bias.append(None)
                self.moving_avg_squared_grad_bias.append(None)
                
    def step(self, debug=False):
        self.bias += 1
        if debug:
            print("Moving avg grad",self.moving_avg_grad)
            print("Moving avg squared grad",self.moving_avg_squared_grad)

                
    def optimize(self, l, neuron, weight, weight_grad):
        if(np.isnan(weight_grad)):
            raise ValueError("WARNING : weight grad is NaN ({})".format((l,neuron)))
        self.moving_avg_grad[l][neuron] = self.beta1*self.moving_avg_grad[l][neuron] + (1-self.beta1)*weight_grad
        if(np.isnan(self.moving_avg_grad[l][neuron])):
            raise ValueError("WARNING : moving avg grad NaN")
        self.moving_avg_squared_grad[l][neuron] = self.beta2*self.moving_avg_squared_grad[l][neuron] + (1-self.beta2)*weight_grad*weight_grad
        if(np.isnan(self.moving_avg_squared_grad[l][neuron])):
            raise ValueError("WARNING : moving avg squared grad NaN")
        m_cap = self.moving_avg_grad[l][neuron]/(1-(self.beta1**self.bias))
        if(np.isnan(m_cap)):
            raise ValueError("bias corrected m_cap is nan")
        v_cap = self.moving_avg_squared_grad[l][neuron]/(1-(self.beta2**self.bias))
        if(np.isnan(v_cap)):
            raise ValueError("bias corrected v_cap is nan")
        
        mod = (self.lr * m_cap)/(np.sqrt(v_cap)+self.epsilon)
        if(np.isnan(mod)):
            raise ValueError("Adam output at t = {} is NaN wtf !".format(self.bias))
        return mod
    
    def optimize_bias(self, l, bias, bias_grad):
        self.moving_avg_grad_bias[l] = self.beta1*self.moving_avg_grad_bias[l] + (1-self.beta1)*bias_grad
        self.moving_avg_squared_grad_bias[l] = self.beta2*self.moving_avg_squared_grad_bias[l] + (1-self.beta2)*bias_grad**2
        m_cap = self.moving_avg_grad_bias[l]/(1-(self.beta1**self.bias))
        v_cap = self.moving_avg_squared_grad_bias[l]/(1-(self.beta2**self.bias))
        return (self.lr * m_cap)/(np.sqrt(v_cap)+self.epsilon)
    
def NoOptimizer(Optimizer):
    def __init__(self, lr=1e-4, lr_grad=1e-6):
        self.lr = lr
        self.lr_grad = lr_grad
    
    def optimize(self, l, neuron, weight, weight_grad):
        return self.lr * weight_grad
    
    def optimize_bias(self, l, bias, bias_grad):
        return self.lr_grad * bias_grad

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
        self.standard = True
        self.name = "Layer"
    
    def out(self, x:np.array) -> np.array:
        pass
    
    def newid(self):
        self.uid = Layer.id_counter
        Layer.id_counter += 1
    
    def init(self, input_size:int, previous_size:int):
        pass
        
    def __repr__(self):
        return "{}(input_size = {}, output_size = {}, activation={}, trainable={})".format(self.name, self.input_size, self.output_size, self.activation.__name__, self.trainable)
    
    def getW(self):
        return self.M

    def get_bias(self):
        return self.bias
    
    def set_weights(self, M, bias=None):
        if M.shape == self.M.shape:
            self.M = M
            if not(bias is None):
                self.bias = bias
        else:
            raise ValueError("New weights do not have the same shape !!! ({} versus {})".format(M.shape, self.getW().shape))
            
    def set_bias(self, bias):
        self.bias = bias
        
    def getNeuronDisabled(self, neuron):
        return False
    
    def step(self, feedforward=False, backpropagation=False):
        pass
    
class Dense(Layer):
    def __init__(self, output_size:int, activation=linear, lr = 1e-3, lr2 = 1e-3, lr_bias=1e-6, dropout=0, bias_only = False, trainable = True, use_bias = True, init_weights = None, init_bias=None):
        Layer.__init__(self, output_size, activation, lr, lr2, lr_bias, trainable, use_bias)
        self.bias = 0
        self.bias_only = bias_only
        self.name = "Dense"
        self.dropout = dropout
        self.dropouts = np.eye(output_size, dtype=floatx)
        if not(init_bias is None):
            self.bias = init_bias
        if not(init_weights is None):
            self.M = init_weights
            
    def init(self, input_size:int, previous_size:int):
        if self.trainable:
            self.M = np.random.normal(0, 1, (input_size, self.output_size)).astype(floatx)
            self.bias = np.random.normal(0, 1)
        self.input_size = input_size
        
    def getW(self):
        return self.M.dot(self.dropouts)
    
    def step(self, feedforward=False, backpropagation=False):
        if feedforward:
            for i in range(self.output_size):
                self.dropouts = np.random.choice([0, 1], p=[self.dropout, 1-self.dropout])
                
            self.dropouts *= (1-self.dropout)
        else:
            self.dropouts = np.eye(self.output_size, dtype=floatx)
    
    def out(self, x:np.array) -> np.array:
                
        if self.bias_only:
            return self.activation(np.ones((1, self.output_size))*self.bias)
        if self.use_bias:
            return self.activation(np.dot(x, self.getW()) + self.bias)
        else:
            return self.activation(np.dot(x, self.getW()))
        
class Flatten(Layer):
    def __init__(self, output_size:int):
        Layer.__init__(self, output_size, trainable=False)
        self.name = "Flatten"
        self.standard = False
        
    def out(self, x:np.array, training=False) -> np.array:
        xnew = x.flatten()
        return xnew.reshape((1, xnew.shape[0]))
    
    def init(self, input_size:int, previous_size:int):
        pass
    
class Dropout(Layer):
    def __init__(self, alpha=0.5, output_size=None):
        Layer.__init__(self, 1, trainable=False)
        self.output_size = output_size
        self.name = "Dropout({})".format(alpha)
        self.alpha = alpha
        self.activation = linear()
        self.standard = False
        raise DeprecationWarning('Dropout layer doesn\'t work well. Please do not use it.')
        
    def init(self, input_size:int, previous_size:int):
        self.output_size = previous_size
        self.input_size = previous_size
        
    def step(self, feedforward=False, backpropagation=False):
        self.M = np.eye(self.input_size, dtype=floatx)
        if backpropagation:
            for i in range(self.M.shape[0]):
                self.M[i,i] = np.random.choice([0, 1], p=[self.alpha, 1-self.alpha])
            self.M *= (1-self.alpha)
            
    def out(self, x:np.array) -> np.array:
        return self.M
    
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
        self.last_epoch = 0
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
            
            self.layers[i].init(isize, self.layers[i-1].output_size)
            
    def summary(self):
        print(self)
        
    def __repr__(self):
        m = "Sequential model (id={}, loss={}) with {} layers :\n".format(self.uid, self.val_loss, len(self.layers))
        for l in self.layers:
            m += str(l) + "\n"
        m += "========\n"
        return m
        
    def predict(self, x:np.array, return_outputs=False) -> np.array:
        if self.debug:
            print("Predict input ",x.shape)
        outputs = [x.copy().flatten()]
        for l in self.layers:
            try:
                x = l.out(x)
                if return_outputs:
                    outputs.append(x.copy().flatten())
            except ValueError as e:
                print(e, " in layer :",l)
                raise(e)
                
        if return_outputs:
            return outputs
        return x.flatten()
    
    def predict_batch(self, x:[np.array], pool=None) -> [np.array]:
        y = []
        if pool is None:
            if self.debug:
                print("No multithreading", file=sys.stderr)
            for el in x:
                y.append(self.predict(el))
        else:
            y = pool.map(self.predict, x)
        return np.array(y)
    
    def get_next_layer(self, l:int):
        lnext = l+1
        while lnext < len(self.layers)-1 and self.layers[lnext].standard == False:
            lnext += 1
        return lnext
    
    @staticmethod
    def _error_calculation(layers, outputs, y, loss):
        try:
            deltas = np.array([np.zeros(layers[l].getW().shape[1]+1) for l in range(len(layers))])
            l = len(layers)
            while l > 0:
                l -= 1
                
                if layers[l].trainable == False:
                    continue
                
                l_next = l+1
                
                for i in range(layers[l].getW().shape[1]):
                    delta = 0
                    if l != len(layers) - 1:
                        for neurone in range(layers[l_next].getW().shape[1]): # Nombre de sorties
                            delta += layers[l_next].getW()[i, neurone] * deltas[l_next][neurone]
                        
                        delta += layers[l_next].bias
                    else:
                        delta = loss.deriv(y[i], outputs[l_next][i]) # On a n plus un outputs (ne pas oublier la couche d'entrée !)
                        
    
                    delta *= layers[l].activation.deriv(outputs[l_next][i])
                    deltas[l][i] += delta
                
                # bias training
                delta = 0
                if layers[l].use_bias:
                    if l != len(layers) - 1:
                        for neurone, val in np.ndenumerate(layers[l_next].getW()): # Here, bias influences **ALL** the next neurons, on all their weights
                            delta += layers[l_next].getW()[neurone] * deltas[l_next][neurone[1]]
                    else:
                        for i in range(len(y)): # Same for outputs
                            delta += loss.deriv(y[i], outputs[l_next][i])
                    
                    for i in range(len(outputs[l_next])):
                        delta *= layers[l].activation.deriv(outputs[l_next][i])
                        
                deltas[l][-1] += delta
                
            return deltas
        except KeyboardInterrupt:
            pass
    
    def train_batch(self, X:[np.array], Y:[np.array], loss:Loss, pool:Pool, optimizer:Optimizer, weight_decay=1e-5):
        
        batch_size = len(X)
        
        for l in self.layers:
            l.step(feedforward=True)
        
        # Forward propagation
        batches_outputs = []
        for step in range(batch_size):
            outputs = self.predict(X[step], return_outputs=True)
            self.losses.append(loss(Y[step], outputs[-1]))
            batches_outputs.append(outputs)
                
        # Backward propagation
        
        # Calcul des erreurs
        for l in range(len(self.layers)):
            self.layers[l].step(backpropagation=True)
        
        batch_deltas = [0]*batch_size
        for step in range(batch_size):
            batch_deltas[step] = pool.apply_async(Sequential._error_calculation, args=(self.layers, batches_outputs[step], Y[step], loss))
            
        deltas = np.array([np.zeros(self.layers[l].getW().shape[1]+1) for l in range(len(self.layers))])
        for step in range(batch_size):
            deltas += batch_deltas[step].get()
            
        # Entrainement
        for l in range(len(self.layers)):
            if self.layers[l].trainable == False:
                continue
            
            for poids, val in np.ndenumerate(self.layers[l].getW()):
                dErr_dweight = - deltas[l][poids[1]]
                if l != len(self.layers) - 1:
                    for step in range(batch_size):
                        dErr_dweight *= batches_outputs[step][l][poids[0]]
                
                if abs(dErr_dweight) > self.clipnorm:
                    dErr_dweight = np.sign(dErr_dweight)*self.clipnorm
                    
                dErr_dweight = optimizer.optimize(l, poids, val, dErr_dweight)
                self.layers[l].getW()[poids] -= dErr_dweight + self.layers[l].lr * weight_decay * self.layers[l].getW()[poids]
            
            if self.layers[l].use_bias:
                dErr_dbias = - deltas[l][-1]
                if abs(dErr_dbias) > self.clipnorm:
                        dErr_dbias = np.sign(dErr_dbias)*self.clipnorm
                
                dErr_dbias = optimizer.optimize_bias(l, self.layers[l].bias, dErr_dbias)
                self.layers[l].bias -= dErr_dbias + self.layers[l].lr * weight_decay * self.layers[l].bias
                
        optimizer.step(self.debug)
        
    def train(self, X:[np.array], Y:[np.array], optimizer:Optimizer, loss=mean_squared_error(), batch_size=1, epochs=100, validation_split=0.1, weight_decay=1e-4, patience=10, lr_decay=1e-3, shuffle=True, resume=True):
        ntrain = int((1-validation_split)*len(X))
        Xtrain = np.array(X[:ntrain])
        Ytrain = np.array(Y[:ntrain])
        Xval = np.array(X[ntrain:])
        Yval = np.array(Y[ntrain:])
        optimizer.init(self.layers)
        pool = Pool(4)
        if not(resume) or self.last_epoch == 0:
            self.best_epoch = 0
            self.counter_since_best = 0
            self.best_layers = copy.deepcopy(self.layers)
            self.history = {"loss":[np.inf], "val_loss":[self.val_loss], "epoch":[0], "lr":[optimizer.lr]}
            resume = 0
        else:
            print("Resuming at epoch {}".format(self.last_epoch+1))
            resume = self.last_epoch
        for epoch in range(resume+1, epochs+1):
            #print(epoch)
            try:
                self.losses = []
                indices = np.arange(Xtrain.shape[0])
                if shuffle:
                    np.random.shuffle(indices)
                num_batch = len(Xtrain) // batch_size
                for batch in range(num_batch):
                    t0 = time.time_ns()
                    index = batch*batch_size
#                    if batch*batch_size >= len(Xtrain):
#                        index = len(Xtrain) - (batch+1)*batch_size
#                        
#                    print(index)
                        
                    excerpt = indices[index:index+batch_size]
                                  
                    self.train_batch(Xtrain[excerpt], Ytrain[excerpt], loss, pool=pool, optimizer=optimizer, weight_decay=weight_decay)
                    
                    if self.debug:
                        time.sleep(3)
                    
                    dt = (time.time_ns() - t0)/1e9
                    
                    print("\r\t => Batch {}/{} - {} batch/s".format(batch+1, num_batch, np.round(1/dt, decimals=3)), end=" "*10)
                
                print("")
                
                train_loss = np.mean(self.losses)
                
                optimizer.lr = (1-lr_decay)*optimizer.lr + lr_decay*optimizer.lr/np.sqrt(epoch)
                optimizer.lr_bias = (1-lr_decay)*optimizer.lr_bias + lr_decay*optimizer.lr_bias/np.sqrt(epoch)
                
                Yval_pred = self.predict_batch(Xval, pool=None)
                self.val_loss = loss(Yval, Yval_pred)
                
                self.history['loss'].append(train_loss)
                self.history["val_loss"].append(self.val_loss)
                self.history['epoch'].append(epoch)
                self.history['lr'].append(optimizer.lr)
                best_message = ""
                if self.val_loss < self.history['val_loss'][self.best_epoch]:
                    best_message = " - ★"
                print("Epoch {} - loss = {}, val_loss = {}{}".format(epoch, train_loss, self.val_loss, best_message))
                
                if self.val_loss < self.history['val_loss'][self.best_epoch]:
                    self.best_epoch = epoch
                    self.counter_since_best = 0
                    self.best_layers = copy.deepcopy(self.layers)
                else:
                    self.counter_since_best += 1
                    #if self.counter_since_best >= 4:
                        #for l in self.layers:
                            #l.lr /= 10
                            #l.lr_bias /= 2
                    if self.counter_since_best >= patience:
                        print("Loss not improving after {} epochs, stopping here".format(self.counter_since_best))
                        print("Restoring best weights from epoch {}...".format(self.best_epoch))
                        self.layers = copy.deepcopy(self.best_layers)
                        Yval_pred = self.predict_batch(Xval, pool=None)
                        self.val_loss = loss(Yval, Yval_pred)
                        print("Loss : {}".format(self.val_loss))
                        break
                    
                self.last_epoch = epoch
            except KeyboardInterrupt: # In case of forced shutdown, we restore the network to its best state
                try:
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    try:
                        pool.terminate()
                    except KeyboardInterrupt:
                        print("Wait a few secs please !")
                    
                print("\nRestoring best weights before shutdown...", file=sys.stderr)
                self.layers = copy.deepcopy(self.best_layers)
                self.last_epoch = self.best_epoch
                self.val_loss = self.history['val_loss'][self.best_epoch]
                del(self.history['val_loss'][self.best_epoch+1:])
                del(self.history['loss'][self.best_epoch+1:])
                del(self.history['lr'][self.best_epoch+1:])
                del(self.history['epoch'][self.best_epoch+1:])
                
                break
        
    def plot_history(self, logarithmic_scale=False):
        import matplotlib.pyplot as plt
        plt.close()
        plt.plot(self.history['epoch'], self.history['loss'], label='loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='val_loss')
        plt.axvline(x=self.best_epoch, linestyle='--', linewidth=0.8, color='red')
        plt.xlabel("Epoch")
        loss_msg = "Loss"
        if logarithmic_scale:
            plt.yscale('log')
            loss_msg += " (log)"
        plt.ylabel(loss_msg)
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
                
                M = layer.getW()
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
                
                
            return model
        
    @staticmethod
    def genetic(models:[Sequential], number:int, bestN:int, lr_qty:float, pool = None, mt = True):
        if pool is None:
            pool = Pool(4)
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