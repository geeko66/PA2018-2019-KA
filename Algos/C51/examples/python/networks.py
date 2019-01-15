#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, \
    Activation, Embedding, Conv2D
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf
#tf.python.control_flow_ops = tf

class Networks(object):

    @staticmethod    
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution

        With States as inputs and output Probability Distributions for all Actions
		
		Karim's notes:
			Thinking about network that just need to maximize time so we will likely use
			some shallow network as a MLP.
		
        """

        """
            My network example:
                state_input = Input(shape=(input_shape))                    # with input_shape = action_size
                hidden_1 = Dense(512, activation='relu')(state_input)
                hidden_2 = ...
                output = distribution_list()

        """

        state_input = Input(shape=(input_shape)) 
        cnn_feature = Conv2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
        cnn_feature = Conv2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
        cnn_feature = Conv2D(64, 3, 3, activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model
