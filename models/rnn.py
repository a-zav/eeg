# -*- coding: utf-8 -*-
"""
  An implementation of the RNN model with GRU units as described in

  An J and Cho S 2016 Hand motion identification of graspand-
  lift task from electroencephalography recordings using
  recurrent neural networks Int. Conf. on Big Data and Smart
  Computing, BigComp 2016 pp 427–9

  The model architecture was adapted to the EEG MI dataset.
"""

import numpy as np

#from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GRU

import util


def classify_rnn(epochs, labels):
    number_of_classes = len(np.unique(labels))

    checkpoint_file = f'{util.CHECKPOINT_DIR}/rnn_{number_of_classes}_classes.h5'

    #X = epochs.get_data()
    #X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3)

    # format: (trials, channels, samples)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels)

    # reshape to (trials, timesteps, features) format expected by the GRU layer
    X_train      = np.swapaxes(X_train, 1, 2)
    X_validate   = np.swapaxes(X_validate, 1, 2)
    X_test       = np.swapaxes(X_test, 1, 2)

    print('X_train shape:', X_train.shape)

    model = create_model((X_train.shape[1], X_train.shape[2]),
                         number_of_classes)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])

    print("Number of parameters:", model.count_params())
    
    checkpointer = ModelCheckpoint(filepath = checkpoint_file,
                                   verbose = 1, save_best_only = True)

    model.fit(X_train, Y_train, batch_size = 16, epochs = 150,
              verbose = 1, validation_data = (X_validate, Y_validate),
              callbacks=[checkpointer],
              class_weight = util.calc_class_weights(labels))

    # load optimal weights
    model.load_weights(checkpoint_file)

    # make prediction on test set
    class_probabilities = model.predict(X_test)
    test_accuracy = util.calc_accuracy_from_prob(class_probabilities, Y_test)
    print("Classification accuracy on the test set: %f " % test_accuracy)
    
    return model, X_test, Y_test

#
# GRU (in: 32 dimensions, out: 32 dimensions)
# -> GRU (in: 32 dimensions, out: 32 dimensions)
# -> Dropout (0.5) ->
# -> Dense(in: 32 dimensions, out: 16 dimensions)
# -> Softmax(in: 16 dimensions, out: number_of_classes=5 dimensions)
#
def create_model(input_data_shape, number_of_classes):
    model = Sequential()
    model.add(Input(shape = input_data_shape))

    layer1 = GRU(32, name="layer1-GRU", return_sequences=True)
    layer2 = GRU(32, name="layer2-GRU")
    layer3 = Dropout(0.5, name="layer2-Dropout")
    layer4 = Dense(16, activation="relu", name="layer3-Dense")
    layer5 = Dense(number_of_classes, activation="softmax", name="layer4-softmax")

    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)

    model.summary()

    return model
