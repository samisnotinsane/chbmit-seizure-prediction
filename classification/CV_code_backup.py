import numpy as np
import mne
from tqdm import tqdm
from ARMA import ARMA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, auc, precision_recall_curve
import matplotlib.pyplot as plt

def load_EEG(filepath, label) -> (np.ndarray, np.ndarray):
    if label == 'preictal':
        data = np.load(filepath)
        target = np.ones(data.shape[1])
    if label == 'interictal':
        data = mne.io \
        .read_raw_edf(input_fname=filepath, preload=False, verbose='Error') \
        .get_data(picks='all', units='uV', return_times=False)
        target = -1
    return data, target

root = '/Volumes/My Passport/AI_Research/data/physionet.org/files/chbmit/1.0.0/' 
case = 'chb01/' 
preictal_filenames = ['chb01_03_preictal.npy', 'chb01_04_preictal.npy', 'chb01_15_preictal.npy', 'chb01_16_preictal.npy',
             'chb01_18_preictal.npy', 'chb01_26_preictal.npy']
interictal_filenames = ['chb01_01.edf', 'chb01_02.edf', 'chb01_05.edf',
                        'chb01_06.edf', 'chb01_07.edf', 'chb01_08.edf']

# load preictal and interictal data
preictal_data_list = []
preictal_target_list = []
interictal_data_list = []
interictal_target_list = []
for i in tqdm(range(6)):
    p_filepath = root + case + preictal_filenames[i]
    ic_filepath = root + case + interictal_filenames[i]
    p_data, p_target = load_EEG(p_filepath, 'preictal')
    ic_data, ic_target = load_EEG(ic_filepath, 'interictal')
    preictal_data_list.append(p_data)
    interictal_data_list.append(ic_data)
    interictal_target_list.append(ic_target)
    preictal_target_list.append(p_target)


# Cross Validation and Shuffle Split
X_preictal = np.array(preictal_data_list)
y_preictal = np.array(preictal_target_list)

X_interictal = np.array(interictal_data_list)
y_interictal = np.array(interictal_target_list)

kf = KFold(n_splits=6)
kf.get_n_splits(X_preictal)
print(kf)

rs = ShuffleSplit(n_splits=len(interictal_filenames), test_size=.80, random_state=0)
rs.get_n_splits(X_interictal)
print(rs)

print('-')
run = 1
for interictal_train_index, interictal_test_index in rs.split(X_interictal, y_interictal):
    print('Run:', run)
    print("Interictal TRAIN:", interictal_train_index, "Interictal TEST:", interictal_test_index)
    print(interictal_train_index[0], interictal_filenames[interictal_train_index[0]], X_interictal[interictal_train_index[0]].shape)
    for interictal_test_idx in interictal_test_index:
        print(interictal_test_idx, interictal_filenames[interictal_test_idx], X_interictal[interictal_test_idx].shape)
    print("---")
    
    for preictal_train_index, preictal_test_index in kf.split(X_preictal, y_preictal):
        print("  Preictal TRAIN:", preictal_train_index, "Preictal TEST:", preictal_test_index)
        for preictal_train_idx in preictal_train_index:
            print('  ', preictal_train_idx, preictal_filenames[preictal_train_idx], X_preictal[preictal_train_idx].shape)
            print("  ---")
    run += 1