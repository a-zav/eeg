# -*- coding: utf-8 -*-
"""
 An implementation of the baseline model:
 EEG raw data -> Wavelet Transform -> SVM classifier
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

import util


def classify_wt_svm(epochs, labels):

    # def wavelet_transform(x):  # x dimensions: (samples, channels, data points)
    #     n_samples, n_channels, n_points = x.shape
    #     #print(x.shape)
    #     out = [] #None
    #     for i in range(0, n_samples):
    #         cA, (cH, cV, cD) = pywt.dwt2(x[i], 'coif1')
    #         print(cA.shape)
    #         all_coeff = np.concatenate([cA, cH, cV, cD], 1)
    #         #all_coeff = all_coeff.reshape(1, all_coeff.shape[0], all_coeff.shape[1])
    #         #print(all_coeff.shape)
    #         out.append(all_coeff)
    #         #out.append(np.concatenate([out, all_coeff], 0))
    #         # if out is None:
    #         #     out = [[all_coeff]]
    #         # else:
    #         #     out.append(all_coeff)
    #             #out = np.concatenate([out, all_coeff], 0)

    #     out = np.asarray(out)
    #     #print(out.shape)
    #     return out.reshape(n_samples, -1)
    
    # def wavelet_transform(x):  # x dimensions: (samples, channels, data points)
    #     n_samples, n_channels, n_points = x.shape
    #     out = None
    #     for ch in range(0, n_channels):
    #         (coef_approx, coef_detail) = pywt.dwt(x[:, ch, :].reshape(-1, n_points), 'coif1')
    #         all_coeff = np.concatenate([coef_approx, coef_detail], 1)
    #         if out is None:
    #             out = all_coeff
    #         else:
    #             out = np.concatenate([out, all_coeff], 1)

    #     print(out.shape)
    #     return out


    def wavelet_transform(x):  # x dimensions: (samples, channels, data points)
        n_samples, n_channels, n_points = x.shape
        print(x.shape)
        x = x.reshape(n_samples, n_channels * n_points)
        coefficients = pywt.dwt(x, 'coif1')
        #coefficients = pywt.wavedec(x, 'coif1', 2)
        #plt.plot(cA.reshape(cA.shape[0], cA.shape[1] * cA.shape[2]))
        #coef_all = np.concatenate([coef_approx, coef_detail], axis=1)
        #coef_all = np.concatenate(list(coefficients), axis=1)
        #print(coef_approx.shape, coef_detail.shape, coef_all.shape)
        #return coef_all.reshape(n_samples, coef_all.shape[1])
        return coefficients[0].reshape(n_samples, coefficients[0].shape[1])


    X = epochs.get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3)

    # build the classifier
    wt = FunctionTransformer(wavelet_transform)
    svc = SVC(kernel='rbf', class_weight='balanced')
    classifier = make_pipeline(wt, svc)

    # do cross-validation on the train data
#    cv = ShuffleSplit(5, test_size=0.3, random_state=1)
#    cv.split(X_train)
#    scores = cross_val_score(classifier, X_train, Y_train, cv=cv, n_jobs=None)

#    print("Cross-validation accuracy: %f" % np.mean(scores))

    # train the model
    model = classifier.fit(X_train, Y_train)

    # make prediction on test set
    predictions = model.predict(X_test)
    test_accuracy = util.calc_accuracy(predictions, Y_test)
    print("Classification accuracy on the test set: %f " % test_accuracy)
    
    return model
