import pickle
import scipy.io as sio
import numpy as np

filename = '../data_files/pickles/GAN/pat_21602_GAN.pickle'
outfile = open(filename, 'rb')
data = pickle.load(outfile)

print(data['GAN'].shape)

mdic = {"GAN_seiz": data['GAN']}
sio.savemat("../results/GAN_seizure_pat_21602.mat", mdic)