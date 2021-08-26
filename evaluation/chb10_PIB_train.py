import numpy as np
from PIB import PIB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import dump

# save path parameters
case = 'chb10'
feature_extract_name = 'PIB'
saveroot = './models/' + case + '/' + feature_extract_name + '/'

# data path parameters
root = './data/Train/' + case
cclass_a = 'Interictal'
cclass_b = 'Preictal'

interictal_files = ['chb10_01_interictal.npy', 'chb10_02_interictal.npy', 'chb10_03_interictal.npy']
preictal_files = ['chb10_12_preictal.npy', 'chb10_20_preictal.npy', 'chb10_27_preictal.npy']

# PIB parameters
bands = [(0.1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 70, 'Low Gamma'), (70, 127.9, 'High Gamma')]
band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
pib = PIB(fs=256, sliding_window=35, bands=bands, band_names=band_names, fft_window='hann', fft_window_duration=20)
print('Neural Rhythms:', bands)

# generate AR features
interictal_feature_list = []
preictal_feature_list = []
for i in range(3):
    # interictal
    filepath = root + '/' + cclass_a + '/' + interictal_files[i]
    data = np.load(filepath)
    features = pib.spin(sig=data)
    interictal_feature_list.append(features)
    # preictal
    filepath = root + '/' + cclass_b + '/' + preictal_files[i]
    data = np.load(filepath)
    features = pib.spin(sig=data)
    preictal_feature_list.append(features)
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
