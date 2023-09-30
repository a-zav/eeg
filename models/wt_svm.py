# -*- coding: utf-8 -*-
"""
 An implementation of the baseline model:
 EEG raw data -> Wavelet Transform -> SVM classifier
"""

import numpy as np
import pywt

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split


def classify_wt_svm(epochs, labels):

    def wavelet_transform(x):  # x dimensions: (samples, channels, data points)
        n_samples, n_channels, n_points = x.shape

        out = None
        for s in range(0, n_samples):
            features = []
            for ch in range(0, n_channels):
                coefficients = pywt.dwt(x[s][ch], 'coif1')
                for c in coefficients:
                    features.extend(c)
            if out is None:
                out = np.empty((n_samples, len(features)))
            out[s] = np.asarray(features)

        return out

    X = epochs.get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3)

    # build the classifier
    wt = FunctionTransformer(wavelet_transform)
    svc = SVC(kernel='rbf', class_weight='balanced')
    classifier = make_pipeline(wt, svc)

    # train the model
    model = classifier.fit(X_train, Y_train)

    return model, X_test, Y_test
