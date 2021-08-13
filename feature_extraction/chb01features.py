import os
import re
import mne
import numpy as np
from ARMA import ARMA
from tqdm import tqdm

root = '/Volumes/My Passport/AI_Research/data/physionet.org/files/chbmit/1.0.0/chb01/'

# filter out files from interictal class
regex = re.compile(r'^(' \
                   'chb01_03\.edf|chb01_04\.edf|chb01_15\.edf|' \
                   'chb01_16\.edf|chb01_18\.edf|chb01_26\.edf|' \
                   '\.(seizures)|\.(txt)|\.(html)' \
                  ')$')
preictal_files = [root+x for x in os.listdir(root) if regex.search(x)]

raws = [mne.io.read_raw_edf(input_fname=x, preload=False, verbose='Error') for x in preictal_files]
print('raws: ', len(raws))

# crop data to preictal interval
raw_crop_1 = raws[0].crop(tmin=2096, tmax=2996)
raw_crop_2 = raws[1].copy().crop(tmin=567, tmax=1467)
raw_crop_3 = raws[2].copy().crop(tmin=832, tmax=1732)
raw_crop_4 = raws[3].copy().crop(tmin=115, tmax=1015)
raw_crop_5 = raws[4].copy().crop(tmin=820, tmax=1720)
raw_crop_6 = raws[5].copy().crop(tmin=962, tmax=1862)

# get data as numpy array
raws_0_EEG, raws_0_times = raw_crop_1.get_data(picks='all', units='uV', return_times=True)
raws_1_EEG, raws_1_times = raw_crop_2.get_data(picks='all', units='uV', return_times=True)
raws_2_EEG, raws_2_times = raw_crop_3.get_data(picks='all', units='uV', return_times=True)
raws_3_EEG, raws_3_times = raw_crop_4.get_data(picks='all', units='uV', return_times=True)
raws_4_EEG, raws_4_times = raw_crop_5.get_data(picks='all', units='uV', return_times=True)
raws_5_EEG, raws_5_times = raw_crop_6.get_data(picks='all', units='uV', return_times=True)

print('EEG 1:', raws_0_EEG.shape, raws_0_times.shape)
print('EEG 2:', raws_1_EEG.shape, raws_1_times.shape)
print('EEG 3:', raws_2_EEG.shape, raws_2_times.shape)
print('EEG 4:', raws_3_EEG.shape, raws_3_times.shape)
print('EEG 5:', raws_4_EEG.shape, raws_4_times.shape)
print('EEG 6:', raws_5_EEG.shape, raws_5_times.shape)

# initialise ARMA parameters
seed = 42
fs = raws[0].info['sfreq'] # assume all data have the same sampling frequency
N = 500
p = 2
m = 5
ar = ARMA(window_width=N, order=p, memory=m, seed=seed)

raws_0_times_ARMA, raws_0_sig_AR, raws_0_sig_MA = ar.spin(sig=raws_0_EEG, fs=fs)
raws_1_times_ARMA, raws_1_sig_AR, raws_1_sig_MA = ar.spin(sig=raws_1_EEG, fs=fs)
raws_2_times_ARMA, raws_2_sig_AR, raws_2_sig_MA = ar.spin(sig=raws_2_EEG, fs=fs)
raws_3_times_ARMA, raws_3_sig_AR, raws_3_sig_MA = ar.spin(sig=raws_3_EEG, fs=fs)
raws_4_times_ARMA, raws_4_sig_AR, raws_4_sig_MA = ar.spin(sig=raws_4_EEG, fs=fs)
raws_5_times_ARMA, raws_5_sig_AR, raws_5_sig_MA = ar.spin(sig=raws_5_EEG, fs=fs)

print('AR 1:', raws_0_sig_AR.shape, raws_0_sig_MA.shape, raws_0_times_ARMA.shape)
print('AR 1:', raws_1_sig_AR.shape, raws_1_sig_MA.shape, raws_1_times_ARMA.shape)
print('AR 1:', raws_2_sig_AR.shape, raws_2_sig_MA.shape, raws_2_times_ARMA.shape)
print('AR 1:', raws_3_sig_AR.shape, raws_3_sig_MA.shape, raws_3_times_ARMA.shape)
print('AR 1:', raws_4_sig_AR.shape, raws_4_sig_MA.shape, raws_4_times_ARMA.shape)
print('AR 1:', raws_5_sig_AR.shape, raws_5_sig_MA.shape, raws_5_times_ARMA.shape)

# save features
np.save('chb01_preictal_features/chb01_preictal_1.npy', raws_0_sig_MA)
print('Saved chb01_preictal_features/chb01_preictal_1.npy')

np.save('chb01_preictal_features/chb01_preictal_2.npy', raws_1_sig_MA)
print('Saved chb01_preictal_features/chb01_preictal_2.npy')

np.save('chb01_preictal_features/chb01_preictal_3.npy', raws_2_sig_MA)
print('Saved chb01_preictal_features/chb01_preictal_3.npy')

np.save('chb01_preictal_features/chb01_preictal_4.npy', raws_3_sig_MA)
print('Saved chb01_preictal_features/chb01_preictal_4.npy')

np.save('chb01_preictal_features/chb01_preictal_5.npy', raws_4_sig_MA)
print('Saved chb01_preictal_features/chb01_preictal_5.npy')

np.save('chb01_preictal_features/chb01_preictal_6.npy', raws_5_sig_MA)
print('Saved chb01_preictal_features/chb01_preictal_6.npy')

# filter out files from preictal class
regex = re.compile(r'^(chb01_03\.edf|chb01_04\.edf|chb01_15\.edf|' \
                   'chb01_16\.edf|chb01_18\.edf|chb01_26\.edf)|\.(seizures)|\.(txt)|\.(html)$')
interictal_files = [root+x for x in os.listdir(root) if not regex.search(x)]

# randomly choose an interictal file
np.random.seed(42)
index = np.random.randint(len(interictal_files))
picked = interictal_files.pop(index)
print('Interictal pick:', picked)
raw = mne.io.read_raw_edf(input_fname=picked, preload=False, verbose='Error')
raw_crop = raw.copy().crop(tmin=0, tmax=2700)

# get data as numpy array
EEG = raw_crop.get_data(picks='all', units='uV', return_times=False)

# ARMA parameters
seed = 42
N = 500
p = 2
m = 30
ar = ARMA(window_width=N, order=p, memory=m, seed=seed)

times_ARMA, sig_AR, sig_MA = ar.spin(sig=EEG, fs=raw_crop.info['sfreq'])

print('AR 1: ', times_ARMA.shape)
print('AR 1: ', sig_AR.shape)
print('AR 1: ', sig_MA.shape)

# save features
np.save('chb01_interictal_features/chb01_interictal_1.npy', sig_MA)
print('Saved chb01_interictal_features/chb01_interictal_1.npy')


# read in remaining interictal files
raws = [mne.io.read_raw_edf(input_fname=x, preload=False, verbose='Error') for x in interictal_files]
print('raws: ', len(raws))

for i in tqdm(range(len(raws))):
    EEG = raws[i].get_data(picks='all', units='uV', return_times=False)
    # ARMA parameters
    seed = 42
    N = 500
    p = 2
    m = 30
    ar = ARMA(window_width=N, order=p, memory=m, seed=seed)
    # On-line ARMA model
    times_ARMA, sig_AR, sig_MA = ar.spin(sig=EEG, fs=raw_crop.info['sfreq'])
#     print(f'AR {i+1}: ', times_ARMA.shape)
#     print(f'AR {i+1}: ', sig_AR.shape)
#     print(f'AR {i+1}: ', sig_MA.shape)
    # save features
    fname = interictal_files[i].split('/')[-1].split('.')[0]
    np.save(f'chb01_interictal_features/{fname}.npy', sig_MA)
# print('Saved chb01_interictal_features/chb01_interictal_1.npy')
    
    
    
