# -*- coding: utf-8 -*-
"""
  An implementation of the new model.
"""

import numpy as np
from typing import NamedTuple

from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation,\
    Dropout, Conv1D, Conv2D, AveragePooling2D, DepthwiseConv2D, SeparableConv1D,\
    SeparableConv2D, BatchNormalization, Flatten, Add, ReLU, MultiHeadAttention,\
    LayerNormalization, DepthwiseConv1D, Reshape, MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers

import util


class Hyperparameters(NamedTuple):
    conv0_filters: int
    pooling_type: str
    pooling_kernel_h: int
    dense1_units: int
    conv1_filters: int
    depthwise_conv1_kernel: int
    attention_heads: int
    residual_fraction: float
    dropout_rate: float
    l2_regularization: float


def classify_new_model(epochs, labels):
    number_of_classes = len(np.unique(labels))

    checkpoint_file =\
        f'{util.CHECKPOINT_DIR}/new_model_{number_of_classes}_classes.h5'

    # format: (trials, channels, samples)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels, scale="range")

    # reshape input data to (trials, channels, samples, kernels=1)
    input_shape = epochs.get_data().shape
    channels, samples, kernels = input_shape[1], input_shape[2], 1
    X_train, X_validate, X_test = util.convert_to_nhwc(
        X_train, X_validate, X_test, channels, samples, kernels)

    hp = Hyperparameters(
        conv0_filters=4,
        pooling_type="avg",
        pooling_kernel_h=8,
        dense1_units=24,
        conv1_filters=1,
        depthwise_conv1_kernel=1,
        attention_heads=2,
        residual_fraction=0.3,
        dropout_rate=0.4,
        l2_regularization=0.0
    )

    model = create_model(channels, samples, number_of_classes, hp)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath = checkpoint_file,
                                   verbose = 1, save_best_only = True)

    model.fit(X_train, Y_train, batch_size=32, epochs=200,
              verbose=2, validation_data=(X_validate, Y_validate),
              callbacks=[
                  checkpointer,
                  EarlyStopping(monitor='val_loss', patience=15),
                  TensorBoard()],
              class_weight=util.calc_class_weights(labels))

    # load optimal weights
    model.load_weights(checkpoint_file)

    return model, X_test, Y_test


def create_feed_forward_layer(last_layer, hp: Hyperparameters):
    lnorm1 = LayerNormalization(axis=1)(last_layer)
    dense1 = Dense(last_layer.shape[1] * 4, activation="swish")(lnorm1)
    dropout1 = Dropout(hp.dropout_rate)(dense1)

    dense2 = Dense(last_layer.shape[1])(dropout1)
    dropout2 = Dropout(hp.dropout_rate)(dense2)
    residual1 = Add()([dropout2, last_layer])

    return residual1


def create_conv_layer(last_layer, hp: Hyperparameters):
    conv_layer = LayerNormalization(axis=1)(last_layer)

    n_filters = hp.conv1_filters
    conv_layer = Conv2D(filters=n_filters, kernel_size=1, padding='same',
                        use_bias=True)(conv_layer)
    
    # conv_layer = DepthwiseConv2D((Chans, 1), use_bias = False, 
    #                          depth_multiplier = D,
    #                          depthwise_constraint = max_norm(1.))(conv_layer)

    conv_layer = Activation('relu')(conv_layer)
    print(conv_layer.shape)

    # conv_layer = Reshape((conv_layer.shape[1] * n_filters,
    #                       conv_layer.shape[2]))(conv_layer)
    conv_layer = Reshape((-1, conv_layer.shape[3]))(conv_layer)
    print(conv_layer.shape)

    # conv_layer = Conv2D(filters=2, kernel_size=(1, 80), padding = 'same',
    #                     use_bias = True, groups=conv_layer.shape[1])(conv_layer)
    conv_layer = DepthwiseConv1D(kernel_size=hp.depthwise_conv1_kernel,
                                 strides=n_filters,
                                 #conv_layer.shape[1]//last_layer.shape[1] // conv_layer.shape[2],
                                 padding='same')(conv_layer)
    print(conv_layer.shape)

    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = Activation('swish')(conv_layer)

#    conv_layer = Reshape((last_layer.shape[1:]))(conv_layer)
    conv_layer = Conv1D(filters=n_filters, kernel_size=1, padding='same',
                        use_bias=True)(conv_layer)

    conv_layer = Dropout(hp.dropout_rate)(conv_layer)
    conv_layer = Reshape((last_layer.shape[1:]))(conv_layer)
    conv_layer = Add()([conv_layer, last_layer])
    #out = swish()(residual1)

    return conv_layer


def create_conformer_block(last_layer, hp: Hyperparameters):
    ff_block = create_feed_forward_layer(last_layer, hp)
    residual1 = Add()([last_layer, ff_block * hp.residual_fraction])
#    residual1 = ReLU()(residual1)
#    residual1 = Reshape((8, 4))(residual1)

    n = residual1.shape[1]
    width = 1

    while n > width and (n % 2 == 0 or n % 3 == 0):
        if n % 2 == 0:
            width *= 2
            n = n // 2
        else:
            width *= 3
            n = n // 3

    if width == 1:
        width = residual1.shape[1]

    residual1 = Reshape((width, -1))(residual1)

    print(f'residual1.shape = {residual1.shape}')
    if hp.attention_heads > 0:
        lnorm1 = LayerNormalization(axis=[1, 2])(residual1)

        n_features = residual1.shape[2]
        key_dim = n_features // hp.attention_heads
        print(f'key_dim = {key_dim}')
        if key_dim == 0:
            key_dim = n_features

        attn1 = MultiHeadAttention(
            num_heads=hp.attention_heads, key_dim=key_dim)(lnorm1, lnorm1)

        dropout1 = Dropout(hp.dropout_rate)(attn1)
        residual2 = Add()([residual1, dropout1])
    else:
        residual2 = residual1

    residual2 = Reshape((width, -1, 1))(residual2)
    conv1 = create_conv_layer(residual2, hp)
    print(f'conv1.shape = {conv1.shape}')

    residual3 = Add()([conv1, residual2])
    residual3 = Flatten()(residual3)
    ff_block2 = create_feed_forward_layer(residual3, hp)
    print(f'ff_block2.shape = {ff_block2.shape}')

    residual4 = Add()([residual3, ff_block2 * hp.residual_fraction])
    lnorm2 = LayerNormalization(axis=1)(residual4)
    flatten1 = Flatten()(lnorm2)

    return flatten1
    

def create_model(channels, samples, number_of_classes, hp: Hyperparameters):
    input1 = Input(shape = (channels, samples, 1))

    if hp.conv0_filters > 0:
        conv1 = Conv2D(hp.conv0_filters, (1, 80), padding = 'same',
                       use_bias = False)(input1)
    else:
        conv1 = input1

    if hp.pooling_type == 'max':
        pooling1 = MaxPooling2D((1, hp.pooling_kernel_h))(conv1)
    elif hp.pooling_type == 'avg':
        pooling1 = AveragePooling2D((1, hp.pooling_kernel_h))(conv1)
    else:
        pooling1 = conv1

    flatten1 = Flatten()(pooling1)
    dense1 = Dense(hp.dense1_units)(flatten1)
    dropout1 = Dropout(hp.dropout_rate)(dense1)

    # a conformer block
    conf1 = create_conformer_block(dropout1, hp)
    #conf1 = create_conformer_block(conf1)

    regularizer = regularizers.L2(hp.l2_regularization) \
        if hp.l2_regularization > 0.0 else None

    dense    = Dense(number_of_classes,
                     activity_regularizer=regularizer)(conf1)
    softmax  = Activation('softmax', name = 'softmax')(dense)
    
    model = Model(inputs=input1, outputs=softmax)
    model.summary()

    return model


def build_trial_model(hp, channels, samples, number_of_classes):

    selected_hp = Hyperparameters(
        conv0_filters = hp.Choice('conv0_filters', values=[0, 2, 4, 8]),
        pooling_type = hp.Choice('pooling_type', values=["max", "avg", "none"]),
        pooling_kernel_h = hp.Choice('pooling_kernel_h', values=[2, 4, 8, 16, 32, 64]),
        dense1_units = hp.Int('dense1_units', min_value=8, max_value=32, step=4),
        conv1_filters = hp.Choice('conv1_filters', values=[1, 2, 4, 8]),
        depthwise_conv1_kernel = \
            hp.Int('depthwise_conv1_kernel', min_value=1, max_value=16, step=1),
        attention_heads = hp.Int('attention_heads', min_value=0, max_value=16, step=1),
        residual_fraction = hp.Float('residual_fraction', 0, 1.0, step=0.1),
        dropout_rate = hp.Float('dropout_rate', 0, 0.5, step=0.1),
        l2_regularization = hp.Float('l2_regularization', 0, 0.3, step=0.1)
    )

    model = create_model(channels, samples, number_of_classes, selected_hp)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])
    
    return model
    