import scipy.io as sio
import numpy as np


mat = sio.loadmat('../results/Results_baseline_noisy_2x.mat')

print(mat['leave_out_result'].shape)
results = mat['leave_out_result']
gmeans = [(results[i][0][0][0][0], "{:.2f}".format(results[i][3][0][0] * 100)) for i in range(30)]

for i in range(30):
    if i not in [1, 6, 7, 16, 29]:
        print(gmeans[i])

