import matplotlib.pyplot as plt
import numpy as np
import utils.data as dt
import scipy.io as sio
from scipy.signal import butter, sosfilt


root_dir = '../'
y, X_GAN = dt.get_data_pickle(root=root_dir, target_patients=['pat_114902'], data_type='non_seiz')
y, X_seiz = dt.get_data_pickle(root=root_dir, target_patients=['pat_102'], data_type='GAN')
X_dict = {'X_GAN': X_seiz}
sio.savemat('../results/pat_102_GAN.mat', X_dict)
exit()

sos = butter(N = 6, Wn = [1/128, 30/128], btype='bandpass', analog=False, output='sos')

X_GAN1 = sosfilt(sos, X_GAN[:, :1024])
X_GAN2 = sosfilt(sos, X_GAN[:, 1024:])
X_seiz1 = sosfilt(sos, X_seiz[:, :1024])
X_seiz2 = sosfilt(sos, X_seiz[:, 1024:])

t = np.linspace(0, 4, 1024)

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.4)
plt.setp(axes, xlim=[0, 4], ylim=[-200, 200], yticks=[-200, 0, 200], xticks=np.arange(5))

axes[0].plot(t, X_seiz1[415])
axes[0].set_ylabel('T3F7', fontsize=12,fontweight="bold")
axes[0].set_xlabel('Time (second)', fontsize=12, fontweight="bold")
axes[0].tick_params(axis='both',  labelsize=12)
axes[1].plot(t, X_seiz2[415])
axes[1].set_ylabel('T4F8', fontsize=12,fontweight="bold")
axes[1].set_xlabel('Time (second)', fontsize=12,fontweight="bold")
axes[1].tick_params(axis='both',  labelsize=12)
# plt.savefig('../outputs/real_4sec.pdf')

# for i in [415, 1534, 1708, 2546, 2850, 3054, 3654]:
for i in [1028, 1088, 1266, 1748, 1736]:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.4)
    plt.setp(axes, xlim=[0, 4], ylim=[-200, 200], yticks=[-200, 0, 200], xticks=np.arange(5))
    print("index: {}".format(i))
    axes[0].plot(t, X_GAN1[i])
    axes[0].set_ylabel('T3F7', fontsize=12, fontweight="bold")
    axes[0].set_xlabel('Time (second)', fontsize=12, fontweight="bold")
    axes[0].tick_params(axis='both', labelsize=12)
    axes[1].plot(t, X_GAN2[i])
    axes[1].set_ylabel('T4F8', fontsize=12, fontweight="bold")
    axes[1].set_xlabel('Time (second)', fontsize=12, fontweight="bold")
    axes[1].tick_params(axis='both', labelsize=12)

    plt.savefig('../outputs/GAN2/figure{}.png'.format(i))
    plt.close()