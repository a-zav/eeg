# -*- coding: utf-8 -*-
"""
 A conformer model implementation. The original paper:

 Gulati, Anmol / Qin, James / Chiu, Chung-Cheng / Parmar, Niki / Zhang, Yu / Yu,
 Jiahui / Han, Wei / Wang, Shibo / Zhang, Zhengdong / Wu, Yonghui / Pang, Ruoming
 Conformer: Convolution-augmented Transformer for Speech Recognition 2020
"""

import numpy as np
from conformer_tf import ConformerConvModule, ConformerBlock

from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint

import util


def classify_conformer(epochs, labels):
    checkpoint_file = f'{util.CHECKPOINT_DIR}/conformer_checkpoint.h5'

    X = epochs.get_data()

    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels)
  
    # reshape input data to (trials, channels, samples, kernels=1)
    channels, samples, kernels = 64, np.shape(X)[2], 1
    X_train, X_validate, X_test = util.convert_to_nhwc(
        X_train, X_validate, X_test, channels, samples, kernels)

    number_of_classes = len(np.unique(labels))

    checkpointer = ModelCheckpoint(filepath=checkpoint_file,
                                   verbose=1, save_best_only=True)

    model = create_model((X_train.shape[1], X_train.shape[2]),
                         number_of_classes)

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


def create_model(input_data_shape, number_of_classes):
    pass
