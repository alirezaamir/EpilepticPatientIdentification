#  MIT License
#
#  Copyright (c)2020 Embedded System Lab, EPFL
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
import numpy as np
import utils.data as dt
import utils.model as model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard
from time import time
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import argparse
import pprint
import json

TEST = 0

# pat_list = ['pat_26102', 'pat_96002', 'pat_16202', 'pat_103002', 'pat_102',
#             'pat_109502', 'pat_85202', 'pat_21602', 'pat_32502', 'pat_92102',
#             'pat_58602', 'pat_111902', 'pat_79502', 'pat_7302', 'pat_93902',
#             'pat_8902', 'pat_30002', 'pat_45402', 'pat_55202', 'pat_23902', 'pat_46702',
#             'pat_75202', 'pat_11002', 'pat_32702', 'pat_21902', 'pat_59102', 'pat_56402',
#             'pat_30802', 'pat_114902', 'pat_22602']

# pat_list_S3 = ["pat_16202", "pat_103002", "pat_109502", "pat_21602", "pat_32502", "pat_92102", "pat_58602",
#                "pat_111902", "pat_93902", "pat_30002", "pat_55202", "pat_23902", "pat_75202", "pat_21902",
#                "pat_56402", "pat_30802"]
val_list = ['pat_16202', 'pat_21902', 'pat_22602', 'pat_55202',  'pat_114902']
pat_list_25 = ['pat_26102', 'pat_96002' , 'pat_103002', 'pat_102',
               'pat_109502', 'pat_85202', 'pat_21602', 'pat_32502', 'pat_92102',
               'pat_58602', 'pat_111902', 'pat_79502', 'pat_7302', 'pat_93902',
               'pat_8902', 'pat_30002', 'pat_45402',  'pat_23902', 'pat_46702',
               'pat_75202', 'pat_11002', 'pat_32702',  'pat_59102', 'pat_56402',
               'pat_30802']

pat_list_2 = ['pat_21902', 'pat_58602']
pat_list_4 = ['pat_75202', 'pat_56402', 'pat_55202', 'pat_45402']
pat_list_8 = ['pat_30002', 'pat_46702', 'pat_75202', 'pat_59102', 'pat_22602', 'pat_23902', 'pat_45402', 'pat_30802']

pat_list = pat_list_25

root_dir = '../'


def train_target_model_tau_S3(target_patients_test):

    y_train, X_train = dt.get_data_pickle(root=root_dir, target_patients=target_patients_test, data_type='non_seiz')
    X_train = dt.get_wavelet(X_train)

    X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)
    X_train, y_train = shuffle(X_train, to_categorical(y_train, len(target_patients_test)))
    # y_test = to_categorical(y_test)

    eeg_model_test = model.get_resnet_model(output_number=len(target_patients_test), dropout=False)

    eeg_model_test.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    eeg_model_test.fit(X_train, y_train, batch_size=64, epochs=40, verbose=0)
    # eeg_model_test.save('outputs/{}_len{}_model.h5'.format(exp_type, len(target_patients_test)))

    return eeg_model_test


def inference_target_model_tau_S3(target_model, exp_type, target_patients_test):
    y_test, X_test = dt.get_data_pickle(root=root_dir, target_patients=target_patients_test, data_type=exp_type)
    # X_test = dt.get_wavelet(X_test)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = to_categorical(y_test)

    eeg_model_test = target_model

    predict = eeg_model_test.predict(X_test)
    predict = np.argmax(predict, axis=1)
    acc = accuracy_score(y_test, to_categorical(predict, len(target_patients_test)))
    return acc


def read_json_results():
    json_file = open("../results/final_results.json")
    return json.load(json_file)


def get_random_patients(num):
    pat_list = shuffle(pat_list_25)
    target_patients = pat_list[:num]


if __name__ == '__main__':
    exp_list = read_json_results()
    for exp in exp_list[:1]:
        target_patients = exp['patients']
        if TEST == 0:
            # Train the models:
            target_model = train_target_model_tau_S3(target_patients)
        else:
            target_model = load_model('outputs/target_model.h5')
        accuracy_seiz = inference_target_model_tau_S3(target_model, 'seiz', target_patients)
        accuracy_GAN = inference_target_model_tau_S3(target_model, 'GAN', target_patients)
        pprint.pprint(target_patients)
        print('original :{}, synthetic: {}'.format(accuracy_seiz, accuracy_GAN))

        exp['norm_orig'] = accuracy_seiz
        exp['norm_synt'] = accuracy_GAN

    with open("../results/norm_results.json") as norm_file:
        json.dump(exp_list, norm_file)
