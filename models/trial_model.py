# -*- coding: utf-8 -*-
"""
  Searches for the best model hyperparameters using a Keras Tuner.
"""

import numpy as np
import keras_tuner as kt

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras.utils import layer_utils

from models.new_model import build_trial_model
import util


class TrialModel(kt.HyperModel):
    MAX_PARAMETERS = 100000

    def __init__(self, channels, samples, number_of_classes):
        self.channels = channels
        self.samples = samples
        self.number_of_classes = number_of_classes
        self.batch_size = None

    def build(self, hp):
        model = build_trial_model(hp, self.channels, self.samples, self.number_of_classes)
        if layer_utils.count_params(model.trainable_weights) > TrialModel.MAX_PARAMETERS:
            raise Exception("The model has too many parameters")
        return model

    def fit(self, hp, model, *args, **kwargs):
        self.batch_size = hp.Choice("batch_size", [16, 32, 64, 128, 256])
        return model.fit(*args, batch_size=self.batch_size, **kwargs)


def find_best_model(epochs, labels):
    number_of_classes = len(np.unique(labels))

    checkpoint_file =\
        f'{util.CHECKPOINT_DIR}/trial_model_{number_of_classes}_classes.h5'
    
    # format: (trials, channels, samples)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test =\
        util.split_data(epochs, labels, scale="range")

    # reshape input data to (trials, channels, samples, kernels=1)
    input_shape = epochs.get_data().shape
    channels, samples, kernels = input_shape[1], input_shape[2], 1
    X_train, X_validate, X_test = util.convert_to_nhwc(
        X_train, X_validate, X_test, channels, samples, kernels)

    tuner = kt.Hyperband(
        TrialModel(channels=channels, samples=samples,
                    number_of_classes=number_of_classes),
        objective='val_accuracy',
        max_epochs=200,
        max_consecutive_failed_trials=20,
        project_name="trial_model",
        overwrite=True
    )

    # We can use RandomSearch instead of Hyperband:
    #
    # tuner = kt.RandomSearch(
    #     TrialModel(channels=channels, samples=samples,
    #                number_of_classes=number_of_classes),
    #     objective='val_accuracy',
    #     max_trials=300,
    #     max_consecutive_failed_trials=20,
    #     project_name="trial_model",
    #     overwrite=True
    # )

    tuner.search_space_summary()

    tuner.search(X_train, Y_train,
                 validation_data=(X_validate, Y_validate),
                 callbacks=[
                     EarlyStopping(monitor='val_loss', patience=15)]
                 )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'>>> Best hp: {best_hp.values}, config: {best_hp.get_config()}')

    model = tuner.hypermodel.build(best_hp)

    class_weights = util.calc_class_weights(labels)

    history = model.fit(X_train, Y_train,
                        epochs=200,
                        verbose=2,
                        validation_data=(X_validate, Y_validate),
                        class_weight=class_weights)

    val_accuracy_history = history.history['val_accuracy']
    best_epoch = val_accuracy_history.index(max(val_accuracy_history)) + 1
    print(f'Best epoch: {best_epoch}')

    checkpointer = ModelCheckpoint(filepath = checkpoint_file,
                                   verbose = 1, save_best_only = True)

    model.fit(X_train, Y_train,
              epochs=best_epoch,
              verbose=2,
              validation_data=(X_validate, Y_validate),
              callbacks=[
                  checkpointer,
                  TensorBoard()],
              class_weight=util.calc_class_weights(labels))

    return model, X_test, Y_test
