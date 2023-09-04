# -*- coding: utf-8 -*-
"""
 A classifier based on the EEGNet model implementation from
 https://github.com/vlawhern/arl-eegmodels
"""

import numpy as np

from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint

import util


def classify_EEGNet(epochs, labels):
    number_of_classes = len(np.unique(labels))

    # saving in the native format ("checkpoint.keras") produces weird errors
    # hence ".h5"
    checkpoint_file = \
        f'{util.CHECKPOINT_DIR}/EEGNet_{number_of_classes}_classes.h5'

    X = epochs.get_data()

    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels, scale="factor")
  
    # reshape input data to (trials, channels, samples, kernels=1)
    channels, samples, kernels = 64, np.shape(X)[2], 1
    X_train, X_validate, X_test = util.convert_to_nhwc(
        X_train, X_validate, X_test, channels, samples, kernels)

    # EEGNet-4,2 model
    model = EEGNet(nb_classes = number_of_classes, Chans = channels,
                   Samples = samples, dropoutRate = 0.5, kernLength = 80,
                   F1 = 4, D = 2, F2 = 8, dropoutType = 'Dropout')
   
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    print("Number of parameters:", model.count_params())
    
    checkpointer = ModelCheckpoint(filepath=checkpoint_file,
                                  verbose=1, save_best_only=True)

    # comment out to use pre-trained weights
    model.fit(X_train, Y_train, batch_size = 16, epochs = 150,
              verbose = 2, validation_data = (X_validate, Y_validate),
              callbacks=[checkpointer],
              class_weight = util.calc_class_weights(labels))
    
    # load optimal weights
    model.load_weights(checkpoint_file)

    return model, X_test, Y_test
