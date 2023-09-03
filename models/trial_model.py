# -*- coding: utf-8 -*-
"""

"""

import random
import numpy as np
import pywt
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Input, Dense, Activation, Dropout, \
    Flatten, SimpleRNN, GaussianNoise

import util


class State(object):
    def __init__(self, last_layer):
        self.n_flops = 0
        self.last_layer = last_layer
        

    def add_required_flops(self, n: int):
        self.n_flops += n


def classify_trial_model(epochs, labels):
    checkpoint_file = f'{util.CHECKPOINT_DIR}/trial_checkpoint.h5'

    number_of_classes = len(np.unique(labels))

    #X = epochs.get_data()
    #X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3)

    # format: (trials, channels, samples)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels)

    rng = random.Random(1)

    model = create_model((X_train.shape[1], X_train.shape[2]),
                         number_of_classes, rng)

    # do cross-validation on the train data
#    cv = ShuffleSplit(5, test_size=0.3, random_state=1)
#    cv.split(X_train)
#    scores = cross_val_score(model, X_train, Y_train, cv=cv, n_jobs=None)
#    print("Cross-validation accuracy: %f" % np.mean(scores))

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])

    print("Number of parameters:", model.count_params())
    
    checkpointer = ModelCheckpoint(filepath = checkpoint_file,
                                   verbose = 1, save_best_only = True)

    model.fit(X_train, Y_train, batch_size = 16, epochs = 150,
              verbose = 2, validation_data = (X_validate, Y_validate),
              callbacks=[checkpointer],
              class_weight = util.calc_class_weights(labels))

    # load optimal weights
    model.load_weights(checkpoint_file)

    # make prediction on test set
    class_probabilities = model.predict(X_test)
    test_accuracy = util.calc_accuracy_from_prob(class_probabilities, Y_test)
    print("Classification accuracy on the test set: %f " % test_accuracy)
    
    return model


def choose_layer(state: State, rng) -> Layer:
    layer_types = ["Dense", "Flatten", "BatchNormalization", "SimpleRNN",
                   "Dropout", "GaussianNoise"]
    layer_type = rng.choice(layer_types)

    layer = None
    if layer_type == "Dense":
        n_units = rng.randint(2, 64)
        activation = rng.choice(["relu", "sigmoid", "tanh", "selu", "elu"])
        layer = Dense(n_units, activation=activation)

    state.nflops += 1000
    state.last_layer = layer

    return layer

#
# Flatten -> Dense(5) -> Softmax:
# 1.3196 - val_accuracy: 0.5595 - 1s/epoch - 2ms/step
# 144/144 [==============================] - 0s 1ms/step
# Classification accuracy on the test set: 0.569063
#
def create_model(input_data_shape, number_of_classes, rng):
    MAX_FLOPS = 10_000

    model = Sequential()
    
    input_layer = Input(shape = input_data_shape)
    model.add(input_layer)

    state: State = State(input_layer)

    while state.n_flops < MAX_FLOPS:
        layer = choose_layer(state, rng)
        model.add(layer)

    #layer1 = Flatten()
    dense = Dense(number_of_classes, activation="relu")
    softmax = Activation('softmax')

    model.add(dense)
    model.add(softmax)

    model.summary()

    return model

