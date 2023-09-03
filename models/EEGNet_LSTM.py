# -*- coding: utf-8 -*-
"""
  An implementation of the EEGNet-LSTM model as described in

  Oliveira, Iago Henrique de and Abner Cardoso Rodrigues (2023). ‘Empirical comparison
  of deep learning methods for EEG decoding’. eng. In: Frontiers in neuroscience
  16, pp. 1003984–1003984. issn: 1662-4548.
"""

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation,\
    Dropout, Conv2D, AveragePooling2D, DepthwiseConv2D,\
    SeparableConv2D, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras import regularizers

import util


def classify_EEGNet_LSTM(epochs, labels):
    checkpoint_file = f'{util.CHECKPOINT_DIR}/EEGNet_LSTM_checkpoint.h5'

    number_of_classes = len(np.unique(labels))

    #X = epochs.get_data()
    #X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3)

    # format: (trials, channels, samples)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels)

    # reshape input data to (trials, channels, samples, kernels=1)
    channels, samples, kernels = 64, np.shape(epochs.get_data())[2], 1
    X_train, X_validate, X_test = util.convert_to_nhwc(
        X_train, X_validate, X_test, channels, samples, kernels)

    model = create_model((X_train.shape[1], X_train.shape[2]),
                         number_of_classes)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    print("Number of parameters:", model.count_params())
    
    checkpointer = ModelCheckpoint(filepath = checkpoint_file,
                                   verbose = 1, save_best_only = True)

    model.fit(X_train, Y_train, batch_size=128, epochs=200,
              verbose=2, validation_data=(X_validate, Y_validate),
              callbacks=[checkpointer],
              class_weight=util.calc_class_weights(labels))

    # load optimal weights
    model.load_weights(checkpoint_file)

    # make prediction on test set
    class_probabilities = model.predict(X_test)
    test_accuracy = util.calc_accuracy_from_prob(class_probabilities, Y_test)
    print("Classification accuracy on the test set: %f " % test_accuracy)
    
    return model

#
# EEGNet Block 1 -> EEGNet Block 2 -> Reshape -> LSTM Layer 1 ->
# Batch Normalisation -> Dropout -> LSTM Layer 2 -> Batch Normalisation ->
# Dropout -> Dense -> Activation (Softmax)
# 
def create_model(input_data_shape, number_of_classes):
    ### EEGnet Block 1 and Block 2
    ### Source: https://github.com/vlawhern/arl-eegmodels
    
    # number of temporal filters to learn
    F1 = 16
    # number of pointwise filters to learn
    F2 = 16
    # number of spatial filters to learn within each temporal convolution
    D = 4
    # length of temporal convolution in first layer
    kernLength = 16

    dropoutRate = 0.2

    # number of channels and time points in the input data
    Chans = input_data_shape[0]
    Samples = input_data_shape[1]

    input1 = Input(shape = (Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding = 'same',
                    input_shape = (Chans, Samples, 1),
                    use_bias = False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias = False, 
                             depth_multiplier = D,
                             depthwise_constraint = max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias = False, padding = 'same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    ### End of EEGNet blocks

    l2_regularizer = regularizers.L2(0.2)

    reshape  = Reshape((32, -1))(block2)
    print(reshape)
    lstm1    = LSTM(32, return_sequences=True)(reshape)
    norm1    = BatchNormalization()(lstm1)
    dropout1 = Dropout(dropoutRate)(norm1)
    lstm2    = LSTM(32)(dropout1)
    norm2    = BatchNormalization()(lstm2)
    dropout2 = Dropout(dropoutRate)(norm2)
    dense    = Dense(number_of_classes,
                     activity_regularizer=l2_regularizer)(dropout2)
    softmax  = Activation('softmax', name = 'softmax')(dense)
    
    model = Model(inputs=input1, outputs=softmax)
    model.summary()

    return model
