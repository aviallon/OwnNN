#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:51:49 2019

@author: aviallon
"""

cpdef float linear(float x)
cpdef float relu(float x)
cpdef float leaky_relu(float x)
cpdef float hard_sigmoid(float x)
cpdef float sigmoid(float x)
cpdef float tanh(float x)

cdef class Layer:
    #cpdef public unsigned id_counter
    cpdef newid(self)
    
cdef class Sequential:
    #cpdef public unsigned id_counter
    cpdef newid(self)
    
#cdef class Evolution:
#    cpdef Sequential _genetic(Sequential model, Sequential parent1, Sequential parent2, float lr_qty)