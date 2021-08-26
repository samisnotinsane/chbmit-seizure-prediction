import numpy as np
from ARMA import ARMA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import dump

# save path parameters
case = 'chb01'
feature_extract_name = 'AR'
saveroot = './models/' + case + '/' + feature_extract_name + '/'

# data path parameters
root = './data/Train/chb01'
cclass_a = 'Interictal'
cclass_b = 'Preictal'

interictal_files = ['chb01_01_interictal.npy', 'chb01_02_interictal.npy', 'chb01_05_interictal.npy']
preictal_files = ['chb01_03_preictal.npy', 'chb01_04_preictal.npy', 'chb01_15_preictal.npy']

# ARMA parameters
seed = 42
fs = 256                 # sampling frequency (Hz)
N = 512                  # fs = N*fp (N must be a natural number)
fp = fs/N                # prediction frequency
n_i = 2                  # AR model order
t_s = 1/fs               # Input signal time period
n_c = 23                 # Number of EEG electrodes (channels)
m = 30                   # MA parameter
print('AR Parameters:')
print(f'Input channels: {n_c}')
print(f'Model: AR({n_i})')
print(f'MA lookback: {m}')
print(f'Window size: {N}')
print(f'Sampling frequency: {fs} Hz')
print(f'Prediction frequency: {fp} Hz')
ar = ARMA(window_width=N, order=n_i, memory=m, seed=seed)

# generate AR features
interictal_feature_list = []
preictal_feature_list = []
for i in range(3):
    # interictal
    filepath = root + '/' + cclass_a + '/' + interictal_files[i]
    data = np.load(filepath)
    _, _, features = ar.spin(sig=data, fs=fs)
    interictal_feature_list.append(features[31:])
    # preictal
    filepath = root + '/' + cclass_b + '/' + preictal_files[i]
    data = np.load(filepath)
    _, _, features = ar.spin(sig=data, fs=fs)
    preictal_feature_list.append(features[31:])
del filepath, data, features

interictal_input = np.vstack(interictal_feature_list)
preictal_input = np.vstack(preictal_feature_list)
print('Interictal input shape:', interictal_input.shape)
print('Preictal input shape:', preictal_input.shape)

interictal_target = -1 * np.ones(interictal_input.shape[0]) # interictal samples are labelled -1
preictal_target = np.ones(preictal_input.shape[0]) # preictal samples are labelled 1

X = np.vstack((interictal_input, preictal_input))
y = np.hstack((interictal_target, preictal_target))
print('X shape:', X.shape)
print('y shape:', y.shape)

# -------

# train model
print('Training model: 1/3')
print('Learner: Support Vector Machine')
print('Kernel: Linear')
svc_linear = SVC(kernel='linear', class_weight='balanced')
print('Learning...', end='')
svc_linear.fit(X, y)
print('[Done]')

# save model
print('Saving model: 1/3')
print('Serialising...', end='')
savepath = saveroot + case + '_' + feature_extract_name + '_' + 'SVM_RBF' + '.joblib'
dump(svc_linear, savepath) 
print('[Done]')
print('Saved: ', savepath)

# -------

# train model
print('Training model: 2/3')
print('Learner: Support Vector Machine')
print('Kernel: Radial Basis Function')
svc_rbf = SVC(kernel='rbf', class_weight='balanced')
print('Learning...', end='')
svc_rbf.fit(X, y)
print('[Done]')

# save model
print('Saving model: 2/3')
print('Serialising...', end='')
savepath = saveroot + case + '_' + feature_extract_name + '_' +  'SVM_Linear' + '.joblib'
dump(svc_rbf, savepath)
print('[Done]')
print('Saved: ', savepath)

# -------

# train model
print('Training model: 3/3')
print('Learner: Logistic Regression')
clf = LogisticRegression(random_state=42)
print('Learning...', end='')
clf.fit(X, y)
print('[Done]')

# save model
print('Saving model: 3/3')
print('Serialising...', end='')
savepath = saveroot  + case + '_' + feature_extract_name + '_' + 'Logistic_Regression' + '.joblib'
dump(clf, savepath)
print('[Done]')
print('Saved: ', savepath)
