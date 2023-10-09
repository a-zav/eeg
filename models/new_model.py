# -*- coding: utf-8 -*-
"""
  An implementation of the new model.
"""

import numpy as np
from typing import NamedTuple

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation,\
    Dropout, Conv1D, Conv2D, AveragePooling2D, \
    LayerNormalization, BatchNormalization, Flatten, Add, MultiHeadAttention,\
    DepthwiseConv1D, MaxPooling2D, LSTM, Permute, Concatenate, TimeDistributed
from keras.optimizers import Adam
from keras import regularizers

import util


class Hyperparameters(NamedTuple):
    use_conformer: bool
    lstm_units: int
    conv0_filters: int
    pooling_type: str
    pooling_kernel_h: int
    dense1_units: int
    conv1_filters: int
    depthwise_conv1_kernel: int
    expansion_factor: int
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
        use_conformer=False,
        lstm_units=32,
        conv0_filters=4,
        pooling_type="avg",
        #pooling_type="max",
        pooling_kernel_h=8,
        dense1_units=16,
        conv1_filters=1,
        depthwise_conv1_kernel=1,
        expansion_factor=4,
        attention_heads=4,
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

    model.fit(X_train, Y_train, batch_size=128, epochs=150,
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
    lnorm1 = LayerNormalization(axis=2)(last_layer)
    dense1 = Dense(last_layer.shape[2] * hp.expansion_factor, activation="swish")
    dense1 = TimeDistributed(dense1)(lnorm1)
    dropout1 = Dropout(hp.dropout_rate)(dense1)

    dense2 = Dense(last_layer.shape[2])
    dense2 = TimeDistributed(dense2)(dropout1)
    dropout2 = Dropout(hp.dropout_rate)(dense2)
    residual1 = Add()([dropout2, last_layer])

    return residual1


def create_conv_layer(last_layer, hp: Hyperparameters):
    conv_layer = LayerNormalization(axis=2)(last_layer)

    n_filters = hp.conv1_filters
    conv_layer = Conv2D(filters=n_filters, kernel_size=1, padding='same',
                        use_bias=True)(conv_layer)
    
    conv_layer = Activation('relu')(conv_layer)

    conv_layer = Reshape((-1, conv_layer.shape[3]))(conv_layer)

    conv_layer = DepthwiseConv1D(kernel_size=hp.depthwise_conv1_kernel,
                                 strides=n_filters,
                                 padding='same')(conv_layer)
    print(conv_layer.shape)

    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = Activation('swish')(conv_layer)

    conv_layer = Conv1D(filters=n_filters, kernel_size=1, padding='same',
                        use_bias=True)(conv_layer)

    conv_layer = Dropout(hp.dropout_rate)(conv_layer)
    conv_layer = Reshape((last_layer.shape[1:]))(conv_layer)
    conv_layer = Add()([conv_layer, last_layer])

    return conv_layer


def create_conformer_block(last_layer, hp: Hyperparameters):
    ff_block = create_feed_forward_layer(last_layer, hp)

    residual1 = Add()([last_layer, ff_block * hp.residual_fraction])

    if hp.attention_heads > 0:
        lnorm1 = LayerNormalization()(residual1)

        n_features = residual1.shape[2]
        key_dim = n_features // hp.attention_heads
        if key_dim == 0:
            key_dim = n_features

        attn1 = MultiHeadAttention(
            num_heads=hp.attention_heads, key_dim=key_dim)(lnorm1, lnorm1)

        dropout1 = Dropout(hp.dropout_rate)(attn1)
        residual2 = Add()([residual1, dropout1])
    else:
        residual2 = residual1

    conv1 = create_conv_layer(residual2, hp)

    residual3 = Add()([conv1, residual2])
    residual3 = Flatten()(residual3)
    ff_block2 = create_feed_forward_layer(residual3, hp)

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

    if hp.lstm_units > 0 or not hp.use_conformer:
        reshape1 = Reshape((channels, -1))(pooling1)
        permute1 = Permute((2, 1))(reshape1)

    if hp.lstm_units > 0:
        lstm1    = LSTM(hp.lstm_units)(permute1)
        lstm1    = LayerNormalization(axis=1)(lstm1)
        lstm1    = Dropout(hp.dropout_rate)(lstm1)

    permute3 = Permute((2, 1, 3))(pooling1)

    if hp.use_conformer:
        dense1 = Dense(hp.dense1_units)
        dense1 = TimeDistributed(dense1)(permute3)
        dropout1 = Dropout(hp.dropout_rate)(dense1)
        conf1 = create_conformer_block(dropout1, hp)
    else:
        num_heads = max(1, hp.attention_heads)
        conf1 = MultiHeadAttention(
            num_heads=num_heads, key_dim=permute3.shape[3] // num_heads)(permute3, permute3)
        conf1 = Flatten()(conf1)
        conf1 = LayerNormalization(axis=1)(conf1)

    if hp.lstm_units > 0:
        conf1 = Concatenate(axis = 1)([conf1, lstm1])

    regularizer = regularizers.L2(hp.l2_regularization) \
        if hp.l2_regularization > 0.0 else None

    dense = Dense(number_of_classes, activity_regularizer=regularizer)(conf1)
    softmax = Activation('softmax', name = 'softmax')(dense)
    
    model = Model(inputs=input1, outputs=softmax)
    model.summary()

    return model


def build_trial_model(hp, channels, samples, number_of_classes):

    selected_hp = Hyperparameters(
        use_conformer = hp.Boolean('use_conformer'),
        lstm_units = hp.Choice('lstm_units', values=[0, 8, 16, 32]),
        conv0_filters = hp.Choice('conv0_filters', values=[0, 2, 4, 8]),
        pooling_type = hp.Choice('pooling_type', values=["max", "avg", "none"]),
        pooling_kernel_h = hp.Choice('pooling_kernel_h', values=[2, 4, 8, 16, 32, 64]),
        dense1_units = hp.Int('dense1_units', min_value=8, max_value=32, step=4),
        conv1_filters = hp.Choice('conv1_filters', values=[1, 2, 4, 8]),
        depthwise_conv1_kernel = \
            hp.Int('depthwise_conv1_kernel', min_value=1, max_value=16, step=1),
        expansion_factor = hp.Int('expansion_factor', min_value=2, max_value=4, step=1),
        attention_heads = hp.Int('attention_heads', min_value=0, max_value=16, step=1),
        residual_fraction = hp.Float('residual_fraction', 0, 1.0, step=0.1),
        dropout_rate = hp.Float('dropout_rate', 0, 0.5, step=0.1),
        l2_regularization = hp.Float('l2_regularization', 0, 0.3, step=0.1)
    )

    model = create_model(channels, samples, number_of_classes, selected_hp)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])
    
    return model
    