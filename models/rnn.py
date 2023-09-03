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
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, \
    Flatten, GRU

import util


def classify_rnn(epochs, labels):
    checkpoint_file = f'{util.CHECKPOINT_DIR}/rnn_checkpoint.h5'

    number_of_classes = len(np.unique(labels))

    #X = epochs.get_data()
    #X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3)

    # format: (trials, channels, samples)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels)

    model = create_model((X_train.shape[1], X_train.shape[2]),
                         number_of_classes)

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

#
# GRU (in: 32 dimensions, out: 32 dimensions) -> Dropout (0.5) ->
# -> Dense(in: 32 dimensions, out: 16 dimensions) ->
# Softmax(in: 16 dimensions, out: number_of_classes=5 dimensions)
#
def create_model(input_data_shape, number_of_classes):
    model = Sequential()
    model.add(Input(shape = input_data_shape))

    layer1 = GRU(32, name="layer1-GRU")
    layer2 = Dropout(0.5, name="layer2-Dropout")
    layer3 = Dense(16, activation="relu", name="layer3-Dense")
    layer4 = Dense(number_of_classes, activation="softmax", name="layer4-softmax")

    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)

    model.summary()

    return model
