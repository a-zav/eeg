# -*- coding: utf-8 -*-
"""
 Miscellaneous utilities.
"""

import os
import numpy as np

from typing import List, Optional
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report,\
    ConfusionMatrixDisplay


# MNE loader uses the following directory structure:
#DATASET_DIR = os.path.dirname(os.path.realpath(__file__)) +\
#    '/../physionet_mi_dataset/MNE-eegbci-data/files/eegmmidb/1.0.0'
DATASET_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../physionet_mi_dataset'

CHECKPOINT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../checkpoints'


def load_data(subjects: list, runs: Optional[List] = None):
    if runs is None:
        runs = list(range(3, 15))

    data_files = []

    for subject in subjects:
        subject_dir = "S{:03d}".format(subject)
        data_files += [f'{DATASET_DIR}/{subject_dir}/{f}'
                       for f in os.listdir(DATASET_DIR + f'/{subject_dir}')
                       if f.endswith('.edf') and any("R{:02d}".format(r) in f for r in runs)]

    eeg_data = [read_raw_edf(f, stim_channel='auto', preload=True) for f in data_files]
    raw = concatenate_raws(eeg_data)
    eegbci.standardize(raw)  # set channel names
    raw.set_montage(make_standard_montage("standard_1005"))
    return raw


def split_data(epochs, labels, scale=None):
    '''
      The following implementation uses fragments of code from
      https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

      'scale' is one of: None (no scaling), "factor" (input data values will
      be multiplied by 1000), "range" (input data will be scaled to have values
      between -1 and 1).

      Returns a tuple of (X_train, Y_train, X_validate, Y_validate, X_test, Y_test)
    '''

    print("np.shape(epochs.get_data()) =", np.shape(epochs.get_data()))
    total_count = np.shape(epochs.get_data())[0]

    indices = np.random.permutation(total_count)

    if scale == 'factor':
        # Scaling raw data by 1000 is suggested at
        # https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
        # to account for scaling sensitivity in deep learning.
        X = np.array([epochs.get_data()[i] * 1000  for i in indices])
    elif scale == 'range':
        X = np.array([epochs.get_data()[i] for i in indices])
        original_shape = X.shape
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X.reshape(original_shape[0], -1))\
            .reshape(original_shape)
    else:
        X = np.array([epochs.get_data()[i] for i in indices])

    y = [labels[i] for i in indices]

    # Split the data into train (50%)/validate (25%)/test (25%).
    n_train = total_count // 2
    test_data_start_index = total_count * 3 // 4

    print(f"Number of trials: {total_count}")
    print(f"train data: [0, {n_train}), " +
          f"validation data: [{n_train}, {test_data_start_index}), " +
          f"test data: [{test_data_start_index}, {total_count-1}]")

    X_train      = X[0:n_train,]
    Y_train      = y[0:n_train]
    X_validate   = X[n_train:test_data_start_index,]
    Y_validate   = y[n_train:test_data_start_index]
    X_test       = X[test_data_start_index:,]
    Y_test       = y[test_data_start_index:]

    # one-hot encode the labels
    Y_train      = np_utils.to_categorical(Y_train)
    Y_validate   = np_utils.to_categorical(Y_validate)
    Y_test       = np_utils.to_categorical(Y_test)

    return (X_train, Y_train, X_validate, Y_validate, X_test, Y_test)


def convert_to_nhwc(X_train, X_validate, X_test, channels, samples, kernels):
    '''
      Converts input data to NHWC format (batch size, height, width, number of channels)
      In our case: (trials, channels, samples, kernels).
      This is needed for CNN models (like EEGNet) implemented in Keras.
      Returns a tuple with converted data: (X_train, X_validate, X_test).
    '''
    print("Number of data points:", samples)
    
    X_train      = X_train.reshape(X_train.shape[0], channels, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], channels, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], channels, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train trials')
    print(X_validate.shape[0], 'validation trials')
    print(X_test.shape[0], 'test trials')
    
    return (X_train, X_validate, X_test)


def calc_class_weights(labels) -> dict:
    '''
      Calculates class weights. Each weight is in [0, 1] range.
      The more examples of the class, the less is the class's weight.
    '''
    __, sample_count = np.unique(labels, return_counts=True)
    total_count = len(labels)
    number_of_classes = len(np.unique(labels))
    class_weights = {}

    for i in range(number_of_classes):
        class_weights[i] = 1 - sample_count[i] / total_count    

    print("Class weights:", class_weights)

    return class_weights


def calc_accuracy_from_prob(class_probabilities, labels, class_names=None):
    predictions = class_probabilities.argmax(axis = -1)
    true_classes = labels.argmax(axis=-1)

    cm = confusion_matrix(true_classes, predictions)
    print("Confusion matrix:", cm)

    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cmd.plot()

    print(classification_report(true_classes, predictions, target_names=class_names))
    return np.mean(predictions == labels.argmax(axis=-1))


def calc_accuracy(predictions, labels, class_names=None):
    cm = confusion_matrix(labels, predictions)
    print("Confusion matrix:", cm)

    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cmd.plot()

    print(classification_report(labels, predictions, target_names=class_names))
    return np.mean(predictions == labels)
