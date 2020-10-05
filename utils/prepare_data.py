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
import os
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter

pat_list = ['pat_102', 'pat_7302', 'pat_8902', 'pat_11002', 'pat_16202', 'pat_21602', 'pat_21902', 'pat_22602',
            'pat_23902', 'pat_26102', 'pat_30002', 'pat_30802', 'pat_32502', 'pat_32702', 'pat_45402', 'pat_46702',
            'pat_55202', 'pat_56402', 'pat_58602', 'pat_59102', 'pat_75202', 'pat_79502', 'pat_85202', 'pat_92102',
            'pat_93902', 'pat_96002', 'pat_103002', 'pat_109502', 'pat_111902', 'pat_114902']


def prepare_non_seiz_data(root=''):
    for pat in pat_list:
        sample_directory = root + '../GAN_epilepsy/GAN_data/Data_to_transform_in_GAN/' + pat
        seiz_files = [os.path.join(sample_directory, sample) for sample in os.listdir(sample_directory) if
                      sample.endswith('.mat')]
        print('num non-seiz files', len(seiz_files))
        input_data = np.zeros((0, 2048))
        selected_number = len(seiz_files) // 20
        for sample_file in seiz_files[:selected_number]:
            mat_content = sio.loadmat(sample_file)
            sample_data = np.array(mat_content['non_seiz'], dtype=np.float32)
            sample_data = np.reshape(sample_data, (1, 2048))
            input_data = np.concatenate((input_data, sample_data))
        filename = 'gen_data/' + pat + '_non_seiz.pickle'
        outfile = open(filename, 'wb')
        out_dict = {'non_seiz': input_data}
        pickle.dump(out_dict, outfile)
        outfile.close()


def prepare_seiz_data(root=''):
    for pat in pat_list:
        sample_directory = root + '../GAN_epilepsy/GAN_data/Data_to_train_GAN/all2/'
        seiz_files = [os.path.join(sample_directory, sample) for sample in os.listdir(sample_directory) if
                              sample.endswith('.mat') and sample.startswith(pat)]
        print('num seiz files', len(seiz_files))
        input_data = np.zeros((0, 2048))
        selected_number = len(seiz_files)//20
        for sample_file in seiz_files[:selected_number]:
            mat_content = sio.loadmat(sample_file)
            sample_data = np.array(mat_content['seiz'], dtype=np.float32)
            sample_data = np.reshape(sample_data, (1, 2048))
            input_data = np.concatenate((input_data, sample_data))
        filename = 'gen_data/' + pat + '_orig.pickle'
        outfile = open(filename, 'wb')
        out_dict = {'orig': input_data}
        pickle.dump(out_dict, outfile)
        outfile.close()


def prepare_non_overlapped_data(root=''):
    for pat in pat_list:
        sample_directory = root + '../GAN_epilepsy/GAN_data/Data_to_train_GAN/all2/'
        seiz_files = [os.path.join(sample_directory, sample) for sample in os.listdir(sample_directory) if
                              sample.endswith('.mat') and sample.startswith(pat)]
        print(seiz_files[:10])
        input_data = np.zeros((0, 2048))
        for i in range(len(seiz_files)):
            sample_file = '../../GAN_epilepsy/GAN_data/Data_to_train_GAN/all2/'+pat+'_GAN_'+str(i)+'.mat'
            # print(sample_file)
            if sample_file not in seiz_files:
                print(sample_file, 'Not found')
                continue
            if i%4 != 0:
                continue
            mat_content = sio.loadmat(sample_file)
            sample_data = np.array(mat_content['seiz'], dtype=np.float32)
            sample_data = np.reshape(sample_data, (1, 2048))
            input_data = np.concatenate((input_data, sample_data))
        filename = pat + '_nol_seiz.pickle'
        outfile = open(filename, 'wb')
        out_dict = {'seiz': input_data}
        pickle.dump(out_dict, outfile)
        print(sorted(seiz_files))


def prepare_GAN_data(root=''):
    for pat in pat_list:
        sample_directory = root + '../GAN_epilepsy/test_set_transformed/sn_b100_l1a_' + pat
        seiz_files = [os.path.join(sample_directory, sample) for sample in os.listdir(sample_directory) if
                              sample.endswith('.mat')]
        input_data = np.zeros((0, 2048))
        selected_number = len(seiz_files) // 20
        for sample_file in seiz_files[:selected_number]:
            mat_content = sio.loadmat(sample_file)
            sample_data = np.array(mat_content['GAN_seiz'], dtype=np.float32)
            sample_data = np.reshape(sample_data, (1, 2048))
            input_data = np.concatenate((input_data, sample_data))
        filename = 'gen_data/' +pat + '_synt.pickle'
        outfile = open(filename, 'wb')
        out_dict = {'synt': input_data}
        pickle.dump(out_dict, outfile)
        outfile.close()


def prepare_non_seiz_close_data(root=''):
    for pat in ['pat_8902', 'pat_11002', 'pat_16202', 'pat_21602', 'pat_21902', 'pat_22602']:
        sample_file = root + '../GAN_epilepsy/GAN_data/Data_to_train_GAN/' + pat +'_labeled_GAN.mat'
        # sample_file = root + '../GAN_epilepsy/GAN_data/Data_to_train_GAN/pat_8902_labeled_GAN.mat'
        mat_content = sio.loadmat(sample_file)
        non_seiz = np.array(mat_content['X_non_seiz'], dtype=np.float32)
        non_seiz_dict = {int(words[0]): words[1:] for words in non_seiz}
        seiz = np.array(mat_content['X_seiz'], dtype=np.float32)
        non_seiz_close = []
        seiz_segments = []
        # print(seiz[:, 0])

        subList = []
        prev_n = -1
        for n in seiz[:, 0]:
            if prev_n + 1 != n:  # end of previous subList and beginning of next
                if subList:  # if subList already has elements
                    seiz_segments.append(subList)
                    subList = []
            subList.append(int(n))
            prev_n = n

        if subList:
            seiz_segments.append(subList)

        first_ptr = 0
        for seg in seiz_segments:
            first = seg[0] -1
            for idx in range(len(seg)):
                if first - idx not in non_seiz_dict.keys():
                    # print(first-idx)
                    # print(non_seiz_dict.keys())
                    break
                non_seiz_close.append(non_seiz_dict[first-idx])
            else:
                if first < seg[0]:
                    first_ptr = first - len(seg)

        for remain_idx in range(seiz.shape[0]- len(non_seiz_close)):
            non_seiz_close.append(non_seiz_dict[first_ptr-remain_idx])

        print(pat,' ptr', first_ptr)
        # for idx in range(seiz.shape[0]):

        print(seiz.shape)
        # print(non_seiz_close)
        print(len(non_seiz_close))
        # sample_data = np.reshape(sample_data, (1, 2048))
        # input_data = np.concatenate((input_data, sample_data))
        filename = pat + '_close.pickle'
        outfile = open(filename, 'wb')
        out_dict = {'close': np.array(non_seiz_close)}
        pickle.dump(out_dict, outfile)
        outfile.close()


def check_data(root=''):
    for pat in pat_list:
        filename = root + pat + '_seiz.pickle'
        outfile = open(filename, 'rb')
        data = pickle.load(outfile)
        print(pat, data['seiz'].shape)


if __name__ == '__main__':
    prepare_GAN_data(root='../')
