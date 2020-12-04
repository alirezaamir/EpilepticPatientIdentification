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

import numpy as np
import scipy.io as sio
import pywt
import os
import pandas as pd
import random
import pickle

pat_list = ['pat_102', 'pat_7302', 'pat_8902', 'pat_11002', 'pat_16202', 'pat_21602', 'pat_21902', 'pat_22602',
            'pat_23902', 'pat_26102', 'pat_30002', 'pat_30802', 'pat_32502', 'pat_32702', 'pat_45402', 'pat_46702',
            'pat_55202', 'pat_56402', 'pat_58602', 'pat_59102', 'pat_75202', 'pat_79502', 'pat_85202', 'pat_92102',
            'pat_93902', 'pat_96002', 'pat_103002', 'pat_109502', 'pat_111902', 'pat_114902']

pat_list_orig = ['pat_102', 'pat_16202', 'pat_11002', 'pat_8902', 'pat_7302', 'pat_26102', 'pat_23902',
                 'pat_22602', 'pat_21902', 'pat_21602',
                 'pat_30002', 'pat_30802', 'pat_32502', 'pat_32702', 'pat_45402', 'pat_46702',
                 'pat_55202', 'pat_56402', 'pat_58602', 'pat_59102',
                 'pat_103002', 'pat_92102', 'pat_93902', 'pat_85202', 'pat_111902', 'pat_75202',
                 'pat_96002', 'pat_79502', 'pat_109502', 'pat_114902']


def get_non_seiz_data(root='', target_patients=None):
    input_data = np.zeros((0, 2048))
    label = np.zeros((0))
    if target_patients is None:
        target_patients = [102, 7302, 8902, 11002]
    for index, pat in enumerate(target_patients):
        sample_file = root + '../GAN_epilepsy/GAN_data/Data_to_transform_in_GAN/pat_' + str(pat) + '_GAN_transform.mat'
        mat_content = sio.loadmat(sample_file)
        seiz = mat_content['X_non_seiz_test']
        size = np.array(seiz).shape[0]
        input_data = np.concatenate((input_data, np.array(seiz)))
        label = np.concatenate((label, np.ones(shape=(size), dtype=int) * index))
    return label, input_data


def get_original_seiz(root='', start_label=0, end_label=30, target_patients=None):
    input_data = np.zeros((0, 2048))
    label = np.zeros((0))
    if target_patients is None:
        target_patients = [102, 7302, 8902, 11002]
    for index, pat in enumerate(target_patients):
        if index < start_label or index > end_label:
            continue
        sample_file = root + '../GAN_epilepsy/GAN_data/Data_to_train_GAN/pat_' + str(pat) + '_GAN.mat'
        print(sample_file)
        mat_content = sio.loadmat(sample_file)
        seiz = mat_content['X_seiz']
        size = np.array(seiz).shape[0]
        input_data = np.concatenate((input_data, np.array(seiz)))
        label = np.concatenate((label, np.ones(shape=(size), dtype=int) * index))
    return label, input_data


def get_synthetic_seiz(root=''):
    input_data = np.zeros((0, 2048))
    label = np.zeros((0))
    for index, pat in enumerate([102, 7302, 8902, 11002]):
        sample_directory = root + '../GAN_epilepsy/test_set_transformed/pat_' + str(pat)
        seiz_files = [os.path.join(sample_directory, sample) for sample in os.listdir(sample_directory) if
                      sample.endswith('.mat')]
        for sample_file in seiz_files:
            mat_content = sio.loadmat(sample_file)
            sample_data = np.array(mat_content['GAN_seiz'], dtype=np.float32)
            sample_data = np.reshape(sample_data, (1, 2048))
            input_data = np.concatenate((input_data, sample_data))
            label = np.concatenate((label, [index]))

    print(input_data.shape)
    print(label.shape)
    filename = 'synthetic.pickle'
    outfile = open(filename, 'wb')
    out_dict = {'data': input_data, 'label': label}
    pickle.dump(out_dict, outfile)
    outfile.close()
    return label, input_data


def get_wavelet(input_data):
    in1 = input_data[:, :1024]
    in2 = input_data[:, 1024:]
    (cA, cD3, cD2, cD1) = pywt.wavedec(in1, 'db4', level=3)
    dwt1 = np.concatenate((cA, cD3, cD2, cD1), axis=1)
    (cA, cD3, cD2, cD1) = pywt.wavedec(in2, 'db4', level=3)
    dwt2 = np.concatenate((cA, cD3, cD2, cD1), axis=1)
    print(input_data.shape)
    print(dwt1.shape)
    input_data = np.concatenate((input_data, dwt1, dwt2), axis=1)
    return input_data


def get_data_features(patient_list, data_list, balance_ratio=None):
    input_data = np.zeros((0, 108))
    label = np.zeros((0))
    for index, dir_name in enumerate(data_list):
        for pat in patient_list:
            sample_file = 'data_files/features/' + dir_name + '/pat_' + str(pat) + '/pat_' + str(
                pat) + '_GAN_' + dir_name + '.mat'
            print(sample_file)
            mat_content = sio.loadmat(sample_file)
            seiz = mat_content['total_features']
            size = np.array(seiz).shape[0]
            df = pd.DataFrame(seiz)  # There are some Nan in the seiz matrix, and they should be removed
            seiz = df.fillna(0)
            if balance_ratio is not None:
                if index == 1:
                    input_data = random.choices(input_data, k=balance_ratio * size)
                    label = label[:balance_ratio * size]
            input_data = np.concatenate((input_data, np.array(seiz, dtype=np.float)))
            label = np.concatenate((label, np.ones(shape=(size), dtype=int) * index))

    return label, input_data


def get_data_pickle(root='', data_type='', target_patients=None):
    if target_patients is None:
        target_patients = pat_list

    input_data = np.zeros((0, 2048))
    label = np.zeros(0)
    for index, pat in enumerate(target_patients):
        filename = root + 'data_files/pickles/' + data_type + '/' + pat + '_' + data_type + '.pickle'
        outfile = open(filename, 'rb')
        data = pickle.load(outfile)
        print(pat, data[data_type].shape)
        size = data[data_type].shape[0]

        input_data = np.concatenate((input_data, data[data_type]))
        label = np.concatenate((label, np.ones(shape=(size), dtype=int) * index))

    return label, input_data
