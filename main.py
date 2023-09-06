# -*- coding: utf-8 -*-
"""
  The driver program.
"""

import os
import random
import numpy as np
#from matplotlib import pyplot as plt
from mne import Epochs, pick_types, events_from_annotations,\
    concatenate_epochs, utils as mne_utils
from mne.io import concatenate_raws
from models.EEGNet import classify_EEGNet
from models.lda import classify_lda
from models.wt_svm import classify_wt_svm
from models.trial_model import classify_trial_model
from models.rnn import classify_rnn
from models.EEGNet_LSTM import classify_EEGNet_LSTM

from streaming import classify_from_stream
from tensorflow import config as tf_config
from tensorflow.keras.utils import set_random_seed

import util


TASK_HANDS_VS_FEET = "hands_vs_feet"
TASK_LEFT_VS_RIGHT = "left_vs_right"

mne_utils.set_config('MNE_USE_CUDA', 'true')
print(f"GPUs: {tf_config.list_physical_devices('GPU')}")


def get_class_names(number_of_classes, task) -> list:
    if number_of_classes <= 3:
        class_names = [] if number_of_classes == 2 else ['Rest']
        class_names += ['Fists', 'Feet'] if task == TASK_HANDS_VS_FEET \
            else ['Left fist', 'Right fist']
    else:
        class_names = [] if number_of_classes == 4 else ['Rest']
        class_names += ['Both fists', 'Both feet', 'Left fist', 'Right fist']

    return class_names


def main():
    # one of: 'WT_SVM', 'LDA', 'RNN', 'EEGNet', 'EEGNet_LSTM', 'TRIAL'
    #model_to_train = 'WT_SVM'
    #model_to_train = 'RNN'
    #model_to_train = 'EEGNet'
    model_to_train = 'EEGNet_LSTM'
    #model_to_train = 'LDA'

    # 2 classes: hands vs feet or left vs right
    # 3 classes: hands vs feet or left vs right plus rest
    # 4 classes: hands vs feet vs left vs right
    # 5 classes: hands vs feet vs left vs right plus rest
    number_of_classes = 3
    # used when number_of_classes is 2 or 3
    #task = TASK_HANDS_VS_FEET
    task = TASK_LEFT_VS_RIGHT

    evaluate_in_streaming_mode = False

    # the following will not help to get reproducible results for Keras models
    # when training on GPU...
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_random_seed(seed)
    rng = random.Random(seed)

    #
    # https://www.physionet.org/content/eegmmidb/1.0.0/
    #
    # The experimental runs:
    #    
    # 1. Baseline, eyes open
    # 2. Baseline, eyes closed
    # 3. Task 1 (open and close left or right fist)
    # 4. Task 2 (imagine opening and closing left or right fist)
    # 5. Task 3 (open and close both fists or both feet)
    # 6. Task 4 (imagine opening and closing both fists or both feet)
    # 7. Task 1
    # 8. Task 2
    # 9. Task 3
    # 10. Task 4
    # 11. Task 1
    # 12. Task 2
    # 13. Task 3
    # 14. Task 4
    #
    # T0 corresponds to rest
    # T1 corresponds to onset of motion (real or imagined) of
    #    the left fist (in runs 3, 4, 7, 8, 11, and 12)
    #    both fists (in runs 5, 6, 9, 10, 13, and 14)
    # T2 corresponds to onset of motion (real or imagined) of
    #    the right fist (in runs 3, 4, 7, 8, 11, and 12)
    #    both feet (in runs 5, 6, 9, 10, 13, and 14)
    #
    
    n_subjects = 109

    # Varsehi and Firoozabadi (2021) and Fan et al. (2021) claim that records
    # of subjects 88, 92, 100 and 104 are damaged, so we excluded them from the analysis.
    excluded_subjects = set([88, 89, 92, 100, 104, 106])
    included_subjects = [i for i in range(1, n_subjects + 1) if i not in excluded_subjects]
    
    left_out_subject = rng.randint(1, len(included_subjects))
    print(f'Will leave subject {left_out_subject} out for testing')
    included_subjects.remove(left_out_subject)
    
    mi_left_vs_right_runs = [4, 8, 12]  # task 2
    mi_hands_vs_feet_runs = [6, 10, 14]  # task 4

    # for testing
    #included_subjects = [1] #, 2, 3]
    #included_subjects = [1, 2, 3, 4, 5]

    # The number of seconds to take before and after the event onset.
    # The models were evaluated using tmax = 0.5 and 3.0 (3.3 for EEGNet_LSTM)
    tmin, tmax = 0.0, 0.5
    #tmin, tmax = 0.0, 3.0

    if number_of_classes > 3 or task == TASK_HANDS_VS_FEET:
        eeg_data_hands_vs_feet = util.load_data(subjects=included_subjects,
                                                runs=mi_hands_vs_feet_runs)

        events_hands_vs_feet_annotated = dict(T1=1, T2=2)
        if number_of_classes == 3 or number_of_classes == 5:
            events_hands_vs_feet_annotated['T0'] = 0

        events_hands_vs_feet, _ = events_from_annotations(
            eeg_data_hands_vs_feet, event_id=events_hands_vs_feet_annotated)

        picks_hands_vs_feet = pick_types(eeg_data_hands_vs_feet.info, meg=False,
                                         eeg=True, stim=False, eog=False, exclude="bads")
    
        #eeg_data_hands_vs_feet = eeg_data.copy().resample(sfreq=128)
        
        #duration_sec = 30
        #for i in range(0, 64):
        #    plt.plot(np.arange(0, duration_sec, 1.0/160.0),
        #             eeg_data_hands_vs_feet.get_data()[i][:160 * duration_sec])

        
        if model_to_train == 'LDA':
            # apply band-pass filter, see
            # https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
            eeg_data_hands_vs_feet.filter(7.0, 40.0, fir_design="firwin",
                                          skip_by_annotation="edge")

        event_id_hands_vs_feet = dict(hands=1, feet=2)
        if number_of_classes == 3 or number_of_classes == 5:
            event_id_hands_vs_feet['rest'] = 0
    
        epochs_hands_vs_feet = Epochs(
            eeg_data_hands_vs_feet,
            events_hands_vs_feet,
            event_id_hands_vs_feet,
            tmin,
            tmax,
            proj=True,
            picks=picks_hands_vs_feet,
            baseline=None,
            preload=True,
        )

    if number_of_classes > 3 or task == TASK_LEFT_VS_RIGHT:
        eeg_data_left_vs_right = util.load_data(subjects=included_subjects,
                                                runs=mi_left_vs_right_runs)
        if model_to_train == 'LDA':
            # apply band-pass filter, see
            # https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
            eeg_data_left_vs_right.filter(1.0, 40.0, fir_design="firwin",
                                          skip_by_annotation="edge")

        event_id_left_vs_right_annotated = dict(T1=3, T2=4)
        if number_of_classes == 3 or number_of_classes == 5:
            event_id_left_vs_right_annotated['T0'] = 0

        events_left_vs_right, _ = events_from_annotations(
            eeg_data_left_vs_right, event_id=event_id_left_vs_right_annotated)

        picks_left_vs_right = pick_types(eeg_data_left_vs_right.info, meg=False,
                                         eeg=True, stim=False, eog=False, exclude="bads")
    
    
        event_id_left_vs_right = dict(left_fist=3, right_fist=4)
        if number_of_classes == 3 or number_of_classes == 5:
            event_id_left_vs_right['rest'] = 0

        epochs_left_vs_right = Epochs(
            eeg_data_left_vs_right,
            events_left_vs_right,
            event_id_left_vs_right,
            tmin,
            tmax,
            proj=True,
            picks=picks_left_vs_right,
            baseline=None,
            preload=True,
        )

    if number_of_classes > 3:
        epochs = concatenate_epochs([epochs_hands_vs_feet, epochs_left_vs_right])
    else:
        epochs = epochs_hands_vs_feet if task == TASK_HANDS_VS_FEET \
            else epochs_left_vs_right

    labels = epochs.events[:, -1]
    if number_of_classes == 3 and task == TASK_LEFT_VS_RIGHT:
        # translate class labels (0, 3, 4) to (0, 1, 2)
        labels -= 2
        labels = np.clip(labels, 0, 2)
    else:
        labels -= np.amin(labels)

    X_test = None
    if model_to_train == 'WT_SVM':
        # Baseline model
        # 102 subjects, task 4: cross-validation classification accuracy:
        #    0.548257 / Chance level: 0.500436
        # 102 subjects, 5 classes:
        #    Cross-validation accuracy: 0.414990
        #    Classification accuracy on the test set: 0.42737
        model, X_test, Y_test = classify_wt_svm(epochs, labels)
    elif model_to_train == 'LDA':
        # 2 classes, hands vs feet:
        # Classification accuracy: 0.585621 / Chance level: 0.500436
        model = classify_lda(epochs, labels)
    elif model_to_train == 'EEGNet':
        # 0-0.5s
        # 102 subjects, task 4: Classification accuracy: 0.707317
        # kernelLength=80, block 1 AveragePooling2D((1, 5)), block2 AveragePooling2D((1, 10)),
        #     Classification accuracy: 0.715157
        # 0-3s blocks: Classification accuracy: 0.713415
        # 0-0.5s, 5 classes: Classification accuracy: 0.596296
        model, X_test, Y_test = classify_EEGNet(epochs, labels)
    elif model_to_train == 'RNN':
        # Classification accuracy on the test set: 0.543355
        model, X_test, Y_test = classify_rnn(epochs, labels)
    elif model_to_train == 'EEGNet_LSTM':
        # Classification accuracy on the test set: 0.631155
        model, X_test, Y_test = classify_EEGNet_LSTM(epochs, labels)
    elif model_to_train == 'TRIAL':
        model, X_test, Y_test = classify_trial_model(epochs, labels)
    else:
        raise Exception(f"Unknown model name '{model}'.")


    if X_test is not None:
        # make prediction on test set and report performance metrics
        predictions = model.predict(X_test)
        class_names = get_class_names(number_of_classes, task)
        if len(Y_test.shape) == 1:
            # Y_test contains numeric class labels, e.g., 0,1,2,3
            test_accuracy = util.calc_accuracy(predictions, Y_test, class_names)
        else:
            # Y_test contains one-hot encoded labels and predictions -
            # the probability of each class
            test_accuracy = util.calc_accuracy_from_prob(predictions,
                                                         Y_test, class_names)

        print("Classification accuracy on the test set: %f " % test_accuracy)


    ###                               ###
    ### Test models in streaming mode ###
    ###                               ###

    if evaluate_in_streaming_mode:
        if number_of_classes > 3:
            eeg_data = concatenate_raws(
                eeg_data_hands_vs_feet, eeg_data_left_vs_right)
        else:
            eeg_data = eeg_data_hands_vs_feet \
                if task == TASK_HANDS_VS_FEET else eeg_data_left_vs_right
        
        classify_from_stream(model, eeg_data)

    return model


if __name__ == '__main__':
    model = main()

print("End.")
