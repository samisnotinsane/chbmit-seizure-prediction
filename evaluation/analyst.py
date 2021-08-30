#!/usr/bin/env python

import os
import click
import yasa
from tqdm import tqdm
import numpy as np
from joblib import load, dump
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

@click.group()
def cli():
    pass

def class_files(patient, sset, class_name, rootdir='./data') -> list:
    root = rootdir + '/' + sset + '/' + patient
    class_a = 'Interictal'
    class_b = 'Preictal'
    if class_name == class_a:
        class_a_dir = root + '/' + class_a
        file_names = [class_a_dir + '/' + x for x in os.listdir(class_a_dir) if not x.startswith('.')]
        print(f'Found {len(file_names)} {class_name} files in: {class_a_dir}')
        return file_names
    if class_name == class_b:
        class_b_dir = root + '/' + class_b
        file_names = [class_b_dir + '/' + x for x in os.listdir(class_b_dir) if not x.startswith('.')]
        print(f'Found {len(file_names)} {class_name} files in: {class_b_dir}')
        return file_names
    return []

def load_data(files) -> list:
    """
    Takes a list of filenames pointing to np arrays then loads and returns those arrays.
    """
    data_list = []
    for i, _ in enumerate(tqdm(files)):
        data = np.load(files[i])
        data_list.append(data)
        del data
    return data_list

def merge_data(array_a, array_b) -> np.ndarray:
    """
    Horizontally merges two arrays with same number of rows.
    """
    hstacked_a = np.hstack(array_a)
    hstacked_b = np.hstack(array_b)
    return np.hstack((hstacked_a, hstacked_b))

def ARMA_core(sig, Ik, n_c, n, n_i, m, mp, model=None, wait_msg=None):
    k_list = []                                                # Sequential buffer for time index in prediction point _k
    a_h_k_list = []                                            # Sequential buffer for AR signal samples
    a_k_list = []                                              # Sequential buffer for ARMA signal samples
    buf_MA = -1 * np.ones((0))                                 # MA prediction signal buffer
    preds = []                                                 # Prediction signal history
    p_MAs = []                                                 # MA prediction signal history
    c = m * np.ones(m)                                         # MA feature coefficients
    c = c / c.sum()
    for _k in tqdm(range(Ik + n_i, n), desc=wait_msg):         # Sliding window
        if (_k % Ik == 0):                                     # Decimation policy: _k occurs once every N samples
            w_start = _k - Ik - n_i + 1                        # Starting index of sliding window (end index is maintained by _k)
            a_h = np.zeros((n_c, n_i))                         # AR parameter estimates from samples in window
            for _i in range(n_c):                              # Iterate channels
                x_t = sig[_i, w_start:_k]                      # Multi-channel window over input signal
                N_w = len(x_t)
                ymat = np.zeros((N_w - n_i, n_i))
                yb = np.zeros((N_w - n_i, n_i))
                for _c in range(n_i, 0, -1):                   # Past dependency of AR up to model order
                    ymat[ : , n_i - _c] = x_t[n_i - _c : -_c]
                yb = x_t[n_i:]
                a_h[_i,:] = np.linalg.pinv(ymat) @ yb            # Least squares solution to optimal parameters via Moore-Penrose Pseudo Inverse
            a_k = np.zeros((n_c, n_i))
            a_h_k_idx = len(a_h_k_list) - 1                    # Index to most recent block of AR parameters of shape: (n_c, n_i)
            for _j in range(m):                                # MA smoothing of AR parameters going back m units of time, in timescale k
                if len(a_h_k_list) > m:                        # Only begin smoothing once unit of time elapsed is greater than m
                    a_k = c[_j] * a_h_k_list[a_h_k_idx - _j]
            a_k = np.mean(a_k, axis=0)                         # Mean over channels
            # classify features (a_k)
            if model != None:
                # p = model.predict(a_k.reshape(1, -1)) 
                p = model.predict(a_k.reshape(1, -1))              # predict based on AR parameters a_k
                preds.append(p)                                    # Add to prediction history
                buf_MA = np.append(buf_MA, p)                      # MA of prediction signal
                p_MA = np.mean(buf_MA)
                if len(buf_MA) == mp:
                    buf_MA = np.delete(buf_MA, 0)
                p_MAs.append(p_MA)                                 # Add to MA prediction history
            k_list.append(_k)                                  # Add to prediction time history
            a_h_k_list.append(a_h)                             # Add reconstructed ARMA history (required by algorithm)
            a_k_list.append(a_k)                               # Add to ARMA response history
            
        # END decimation condition
    # END sliding window loop
    return k_list, a_k_list, preds, p_MAs

def ARMA(sig, fs=None, model=None, wait_msg='Analysing... '):
    # initialise ARMA parameters
    fs = 256           # sampling frequency in Hz
    window = 512       # window width
    order = 2          # Second order model AR(2)
    feature_mem = 1    # MA smoothing for AR features
    predict_mem = 10   # MA smoothing for prediction signal
    fp = fs/window     # Prediction frequency
    print(f'ARMA{order, feature_mem}')
    print(f'Window size: {window}')
    print(f'Sampling frequency: {fs} Hz')
    print(f'Prediction frequency: {fp} Hz')
    print(f'AR parameter smoothing: {feature_mem}')
    print(f'Prediction smoothing: {predict_mem}')

    n_c = sig.shape[0] # channel count in EEG
    n = sig.shape[1] #  sample count in EEG
    _ks, a_h_k, p_k, p_MA_k = ARMA_core(sig, window, n_c, n, order, feature_mem, predict_mem, model, wait_msg)
    times = np.hstack(_ks)
    response = np.vstack(a_h_k)
    if len(p_k) > 0:
        prediction = np.hstack(p_k)
        prediction_MA = np.hstack(p_MA_k)
    else:
        prediction = np.zeros_like(times)
        prediction_MA = np.zeros_like(times)
    return times, response, prediction, prediction_MA

def write_response_plot(times, response, preictal_start_time, savename, saveto, saveformat, x_lim_end=3.75) -> None:
    print('Generating plot...', end='')
    Path(saveto).mkdir(parents=True, exist_ok=True) # create saveto directory if not exists
    savepath = saveto + '/' + savename + saveformat
    sns.set_palette(sns.color_palette('Set2'))
    plt.figure(figsize=(12,6))
    sns.lineplot(x=times, y=response[:,0], label='$a_1$')
    ax = sns.lineplot(x=times, y=response[:,1], label='$a_2$')
    if preictal_start_time != -1:
        ax.fill_between(times, 0, 1, where=times < preictal_start_time, color='#9cd34a', alpha=0.3, transform=ax.get_xaxis_transform(), label='Interictal')
        ax.fill_between(times, 0, 1, where=times > preictal_start_time, color='#ffd429', alpha=0.3, transform=ax.get_xaxis_transform(), label='Preictal')

    plt.xticks(np.arange(0,3.76,0.25))
    plt.xlim([0,x_lim_end])
    # plt.ylim([-0.2,0.2])
    plt.xlabel('Time, $h$')
    plt.ylabel('A.U.')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(savepath)
    click.secho('[Done]', fg='green')
    click.secho(f'Response plot saved to: {savepath}')

def write_spectral_response_plot(times, response, preictal_start_time, savename, saveto, saveformat='.pdf', x_lim_end=3.75):
    print('Generating plot...', end='')
    Path(saveto).mkdir(parents=True, exist_ok=True) # create saveto directory if not exists
    savepath = saveto + '/' + savename + saveformat
    delta = response[:,0] # parse array into named bands
    theta = response[:,1]
    alpha = response[:,2]
    beta = response[:,3]
    low_gamma = response[:,4]
    high_gamma = response[:,5]

    sns.set_palette(sns.color_palette('Set2'))
    plt.figure(figsize=(12,6))
    sns.lineplot(x=times, y=delta, label=r'$\delta$')
    sns.lineplot(x=times, y=theta, label=r'$\theta$')
    sns.lineplot(x=times, y=alpha, label=r'$\alpha$')
    sns.lineplot(x=times, y=beta, label=r'$\beta$')
    sns.lineplot(x=times, y=low_gamma, label=r'$\gamma$')
    ax = sns.lineplot(x=times, y=high_gamma, label=r'$\Gamma$')
    if preictal_start_time != -1:
        ax.fill_between(times, 0, 1, where=times < preictal_start_time, color='#9cd34a', alpha=0.3, transform=ax.get_xaxis_transform(), label='Interictal')
        ax.fill_between(times, 0, 1, where=times > preictal_start_time, color='#ffd429', alpha=0.3, transform=ax.get_xaxis_transform(), label='Preictal')
    plt.xticks(np.arange(0,3.76,0.25))
    plt.xlim([0,x_lim_end])
    plt.ylim([0,1])
    plt.xlabel('Time, $h$')
    plt.ylabel('Relative Bandpower')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(savepath)
    click.secho('[Done]', fg='green')
    click.secho(f'Response plot saved to: {savepath}')

def write_prediction_plot(times, prediction, MA_prediction, preictal_start_time, model_name, savename, saveto, saveformat, x_lim_end=3.75, alarm_threshold=True) -> None:
    print('Generating plot...', end='')
    Path(saveto).mkdir(parents=True, exist_ok=True) # create saveto directory if not exists
    savepath = saveto + '/' + savename + saveformat
    sns.set_palette(sns.color_palette('Set2'))
    plt.figure(figsize=(12,6))
    ax = sns.lineplot(x=times, y=prediction, label=model_name)
    sns.lineplot(x=times, y=MA_prediction, label='MA')
    if alarm_threshold:
        ax.axhline(y=0, ls='--', color='k', label='Alarm Threshold')
    ax.fill_between(times, 0, 1, where=times < preictal_start_time, color='#9cd34a', alpha=0.3, transform=ax.get_xaxis_transform(), label='Interictal')
    ax.fill_between(times, 0, 1, where=times > preictal_start_time, color='#ffd429', alpha=0.3, transform=ax.get_xaxis_transform(), label='Preictal')

    plt.xticks(np.arange(0, x_lim_end+0.01, 0.25))
    tickpos = [-1,1] # show only classes: -1 and 1
    plt.yticks(tickpos,tickpos)
    plt.xlim([0,x_lim_end])
    plt.ylim([-2,2])
    plt.xlabel('Time, $h$')
    plt.ylabel('Class')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(savepath)
    click.secho('[Done]', fg='green')
    click.secho(f'Prediction plot saved to: {savepath}')

def write_ARMA_jointplot(X, y, savename, saveto, saveformat) -> None:
    print('Generating plot...', end='')
    savepath = saveto + '/' + savename + saveformat
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Period'] = y
    mapping = {-1: "Interictal", 1: "Preictal"} # map numeric target to string label
    df.replace({"Period": mapping}, inplace=True)
    palette = sns.color_palette('Set2', n_colors=2)
    sns.jointplot(data=df, x='Feature 1', y='Feature 2', hue='Period', kind='kde', palette=palette, alpha=0.6)
    plt.tight_layout()
    plt.savefig(savepath)
    click.secho('[Done]', fg='green')
    click.secho(f'ARMA jointplot saved to: {savepath}')

def write_spectral_pairplot(X, y, band_names, savename, saveto, saveformat='.pdf') -> None:
    print('Generating plot...', end='')
    savepath = saveto + '/' + savename + saveformat
    df = pd.DataFrame(X, columns=band_names)
    df['Period'] = y
    di = {-1: "Interictal", 1: "Preictal"} # map numeric target to string label
    df.replace({"Period": di}, inplace=True)
    palette = sns.color_palette("Set2", n_colors=2)
    sns.pairplot(df, hue='Period', height=1.8, aspect=1.8, palette=palette,
                    plot_kws=dict(edgecolor="k", linewidth=0.5),
                    diag_kind="kde", diag_kws=dict(shade=True))
    plt.tight_layout()
    plt.savefig(savepath)
    click.secho('[Done]', fg='green')
    click.secho(f'Spectral pairplot saved to: {savepath}')

def learn_with_and_remember(X, y, model_name, savename, saveto):
    print('Serialising model...', end='')
    if model_name == 'Linear SVM':
            # click.secho('Training Linear Kernel SVM', fg='blue')
            model = SVC(kernel='linear', class_weight='balanced')
    if model_name == 'RBF SVM':
        # click.secho('Training Radial Basis Function Kernel SVM', fg='blue')
        model = SVC(kernel='rbf', class_weight='balanced')
    if model_name == 'Logistic Regression':
        # click.secho('Training Logistic Regression', fg='blue')
        model = LogisticRegression(random_state=42)
    model.fit(X, y)
    saveformat = '.joblib'
    savepath = saveto + '/' + savename + saveformat
    dump(model, savepath)
    click.secho('[Done]', fg='green')
    click.secho(f'Model saved to: {savepath}')

def load_learner(models, patient, method, learner):
    print('Loading model...', end='')
    fileformat = '.joblib'
    if method.lower() == 'spectral':
        modelpath = models + '/' + patient + '/' + method + '/' + learner + fileformat
    else:
        modelpath = models + '/' + patient + '/' + method[0:2] + '/' + learner + fileformat # assumes directory in patient folder is named 'AR'
    model = load(modelpath)
    click.secho('[Done]', fg='green')
    return model

def make_learner_name_human_readable(learner_name):
    kernel_name = learner_name.split('_')[2]
    model_name = learner_name.split('_')[3]
    return kernel_name + ' ' + model_name

def get_neural_rhythm_bands():
    bands = [(0.1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 70, 'Low Gamma'), (70, 127.9, 'High Gamma')]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
    return bands, band_names

def neural_power_core(sig, fs, bands, band_names, online_window_size=2, fft_window_size=2, fft_window_name='hann', model=None, MA_smoothing=10, wait_msg='Analysing... '):
    N = fs*online_window_size
    n = sig.shape[1]
    fp = fs/N

    print('Spectral configuration:')
    print(f'Input length: {n}')
    print(f'Input channels: {sig.shape[0]}')
    print(f'Sampling frequency: {fs} Hz')
    print(f'Prediction frequency: {fp} Hz')
    print(f'Stream buffer size: {online_window_size}')
    print(f'FFT window: {fft_window_name}')
    print(f'FFT window size: {fft_window_size}')
    print(f'Spectral bands: {bands}')
    print(f'Prediction smoothing: {MA_smoothing}')

    time_list = [] # prediction time history
    bandpower_list = [] # bandpower history
    predict_list = [] # prediction history
    buf_MA = -1 * np.ones((0))   # MA prediction signal buffer
    p_MAs = []
    Ik = N
    for _k in tqdm(range(Ik, n), desc=wait_msg):
        if(_k % N == 0):
            w_start = _k - Ik
            x_t = sig[:, w_start:_k]
            df = yasa.bandpower(x_t, sf=fs, win_sec=fft_window_size, bands=bands, bandpass=True, relative=True, kwargs_welch={'window': fft_window_name})
            data = df[band_names].to_numpy()
            data_mean = np.mean(data, axis=0)
            if model != None:
                prediction = model.predict(data_mean.reshape(1, -1))      # Model prediction
                predict_list.append(prediction)
                buf_MA = np.append(buf_MA, prediction)                      # MA of prediction signal
                p_MA = np.mean(buf_MA)
                if len(buf_MA) == MA_smoothing:
                    buf_MA = np.delete(buf_MA, 0)
                p_MAs.append(p_MA)                                 # Add to MA prediction history
            time_list.append(_k)
            bandpower_list.append(data_mean)
    prediction_times = np.hstack(time_list)
    times = np.arange(0, prediction_times.shape[0]) / (fs/N) / 3600   # predict signal times (h)
    bandpowers = np.vstack(bandpower_list)
    if model != None:
        prediction = np.hstack(predict_list)
        prediction_MA = np.hstack(p_MAs)
    else:
        prediction = np.zeros_like(times)
        prediction_MA = np.zeros_like(times)
    return times, bandpowers, prediction, prediction_MA

def create_X_y(class_a_response, class_b_response):
    print('Creating X, y...', end='')
    # target labels
    class_a_targets = -1 * np.ones(class_a_response.shape[0])
    class_b_targets = np.ones(class_b_response.shape[0])
    # merge input classes
    X = np.vstack((class_a_response, class_b_response))
    # merge target classes
    y = np.hstack((class_a_targets, class_b_targets))
    click.secho('[Done]', fg='green')
    print(f'Merged input dimensionality: {X.shape}')
    print(f'Merged target dimensionality: {y.shape}')
    return X, y

@cli.command()
@click.option('--patient', required=True, help='Patient identifier (e.g. \'chb01\')')
@click.option('--method', required=True, help='Feature extraction method. Choose either: \'ARMA\'  or \'Spectral\'')
@click.option('--learner', required=True, help='Path to trained model')
@click.option('--train', required=True, is_flag=True, help='Runs on training data if supplied. Otherwise runs on test data')
@click.option('--data', required=True, help='Root directory of train test data.')
@click.option('--models', required=True, help='Root directory of models.')
@click.option('--saveto', required=True, help='Path to directory where generated plots will be saved.')
@click.option('--saveformat', help='File format of plot (e.g. .png, .pdf)')
@click.option('--debug', is_flag=True, help='Uses smaller portion of data for quick runs')
def think(patient, method, learner, train, data, models, saveto, saveformat, debug):
    """
    Predict seizure from EEG data in real time.
    """
    Path(saveto).mkdir(parents=True, exist_ok=True) # create saveto directory if not exists
    learner_name = (learner.split('/')[-1]).split('.')[0]
    human_readable_learner_name = make_learner_name_human_readable(learner_name)
    if train:
        sset = 'Train'
        x_lim_end = 1.5
        response_savename = f'{method}_response_TRAIN'
        prediction_savename = f'Prediction_{learner_name}_MA_TRAIN'
        alarm_threshold = False
    else:
        sset = 'Test'
        x_lim_end = 3.75
        response_savename = f'{method}_response_TEST'
        prediction_savename = f'Prediction_{learner_name}_MA_TEST'
        alarm_threshold = True
    # load data
    click.echo(f'Dataset: {sset}')
    class_a_name = 'Interictal'
    class_b_name = 'Preictal'
    class_a_files = class_files(patient, sset, class_a_name)
    class_b_files = class_files(patient, sset, class_b_name)
    
    if debug:
        # load less data in debug mode
        class_a_files = class_a_files[0:1]
        class_b_files = class_b_files[0:1]

    click.echo(f'Loading {len(class_a_files)} {class_a_name} arrays')
    class_a_data = load_data(class_a_files)
    click.echo(f'Loading {len(class_a_files)} {class_b_name} arrays')
    class_b_data = load_data(class_b_files)

    X = merge_data(class_a_data, class_b_data)
    click.secho(f'Loaded merged data: {X.shape}')

    model = load_learner(models, patient, method, learner)
    print(f'Model: {model}')
    click.secho(f'Hyperparameters:\n {model.get_params()}')

    if method == 'ARMA':
        click.secho(f'Begin real-time prediction with model \'{learner}\' on patient \'{patient}\' using \'{method}\'.', fg='blue')
        # initialise ARMA parameters
        # online prediction
        fs = 256
        window = 512
        times, response, prediction, MA_prediction = ARMA(X, fs, model)
        print('times:', times.shape)
        print('prediction:', prediction.shape)
        print('MA_prediction:', MA_prediction.shape)
        
        class_a_data_hstacked = np.hstack(class_a_data)
        times_in_hour = np.arange(0, times.shape[0]) / (fs/window) / 3600
        preictal_start_time = np.rint(np.max( (np.arange(0, class_a_data_hstacked.shape[1]) / fs) )) / 3600
        print('Preictal start (h):', preictal_start_time)

        # plots
        write_response_plot(times_in_hour, response, preictal_start_time, response_savename, saveto, saveformat, x_lim_end=x_lim_end)
        write_prediction_plot(times_in_hour, prediction, MA_prediction, preictal_start_time, human_readable_learner_name, prediction_savename, saveto, saveformat, x_lim_end=x_lim_end, alarm_threshold=alarm_threshold)

    if method.lower() == 'spectral':
        click.secho(f'Begin real-time prediction with model \'{learner}\' on patient \'{patient}\' using \'{method}\'.', fg='magenta')
        online_window_size = 35 # WARNING: Do not change value without knowing what you're doing! Increasing value will not produce features for full time length. Decreasing will distort signal.
        fs = 256
        bands, band_names = get_neural_rhythm_bands()
        times, response, prediction, prediction_MA = neural_power_core(sig=X, fs=fs, bands=bands, band_names=band_names, online_window_size=online_window_size, fft_window_size=2, fft_window_name='hann', model=model, MA_smoothing=10)
        print('Dimensionality of times:', times.shape)
        print('Dimensionality of response:', response.shape)
        print('Dimensionality of prediction:', prediction.shape)
        print('Dimensionality of prediction MA:', prediction_MA.shape)

        class_a_data_hstacked = np.hstack(class_a_data)
        times_in_hour = np.arange(0, times.shape[0]) / (fs/(fs*online_window_size)) / 3600
        preictal_start_time = np.rint(np.max( (np.arange(0, class_a_data_hstacked.shape[1]) / fs) )) / 3600
        print('Preictal start (h):', preictal_start_time)

        # plots
        write_spectral_response_plot(times_in_hour, response, preictal_start_time, response_savename, saveto, x_lim_end=1.50)
        write_prediction_plot(times_in_hour, prediction, prediction_MA, preictal_start_time, human_readable_learner_name, prediction_savename, saveto, saveformat, x_lim_end=x_lim_end, alarm_threshold=alarm_threshold)
    click.secho(f'Completed online prediction with model \'{learner}\' on patient \'{patient}\' using \'{method}\'.', fg='green')

@cli.command()
@click.option('--patient', required=True, help='Patient identifier (e.g. \'chb01\')')
@click.option('--method', required=True, help='Feature extraction method. Choose either: \'ARMA\'  or \'Spectral\'')
@click.option('--learning_algorithm', required=True, help='Machine learning algorithm for training. Choose either: \'Linear SVM\', \'RBF SVM\', \'Logistic Regression\'')
@click.option('--data', required=True, help='Root directory of train test data.')
@click.option('--learnersaveto', required=True, help='Path to directory where trained model will be saved.')
@click.option('--plot_figures', is_flag=True, help='Root directory of train test data.')
def teach(patient, method, learning_algorithm, data, learnersaveto, plot_figures):
    Path(learnersaveto).mkdir(parents=True, exist_ok=True) # create saveto directory if not exists
    sset = 'Train' # use training set for model training
    # load data
    click.echo(f'Dataset: {sset}')
    class_a_name = 'Interictal'
    class_b_name = 'Preictal'
    class_a_files = class_files(patient, sset, class_a_name, rootdir=data)
    class_b_files = class_files(patient, sset, class_b_name, rootdir=data)
    class_a_data = load_data(class_a_files)
    class_b_data = load_data(class_b_files)
    click.echo(f'Loaded {len(class_a_files)} {class_a_name} arrays')
    click.echo(f'Loaded {len(class_a_files)} {class_b_name} arrays')
    
    print(f'Dimensionality of single array in {class_a_name} class:', class_a_data[0].shape)
    print(f'Dimensionality of single array in {class_b_name} class:', class_b_data[0].shape)
    class_a_data_hstacked = np.hstack(class_a_data)
    class_b_data_hstacked = np.hstack(class_b_data)
    print(f'Dimensionality of {len(class_a_files)} {class_a_name} class arrays: {class_a_data_hstacked.shape}')
    print(f'Dimensionality of {len(class_b_files)} {class_b_name} class arrays: {class_b_data_hstacked.shape}')

    # generate features
    if method.upper() == 'ARMA':
        click.secho(f'Begin teaching {learning_algorithm} about patient {patient} using {method}.', fg='blue')
        click.secho(f'Generating features for class: {class_a_name}')
        print(f'Input channels {class_a_name}: {class_a_data_hstacked.shape[0]}')
        print(f'Input length {class_a_name}: {class_a_data_hstacked.shape[1]}')
        wait_msg = 'Extracting features... '
        _, class_a_response, _, _ = ARMA(class_a_data_hstacked, wait_msg=wait_msg)
        print(f'{class_a_name} response dimensionality: {class_a_response.shape}')

        click.secho(f'Generating features for class: {class_b_name}')
        print(f'Input channels {class_b_name}: {class_b_data_hstacked.shape[0]}')
        print(f'Input length {class_b_name}: {class_b_data_hstacked.shape[1]}')
        times, class_b_response, _, _ = ARMA(class_b_data_hstacked, wait_msg=wait_msg)
        print(f'{class_b_name} response dimensionality: {class_b_response.shape}')
        print(f'{method} times dimensionality: {times.shape}')

        # AR response plots
        if plot_figures:
            saveto = f'./figures/{patient}/AR'
            class_a_savename = 'train_class_a_response'
            class_b_savename = 'train_class_b_response'
            saveformat = '.pdf'
            preictal_start_time = -1
            fs = 256
            window = 512
            times_in_hour = np.arange(0, times.shape[0]) / (fs/window) / 3600
            write_response_plot(times_in_hour, class_a_response, preictal_start_time, class_a_savename, saveto, saveformat, x_lim_end=0.75)
            write_response_plot(times_in_hour, class_b_response, preictal_start_time, class_b_savename, saveto, saveformat, x_lim_end=0.75)

        X, y = create_X_y(class_a_response, class_b_response)

        # visualise ARMA feature distribution
        if plot_figures:
            savename = 'train_jointplot'
            write_ARMA_jointplot(X, y, savename, saveto, saveformat)

        # train and save model
        model_name = learning_algorithm.split(' ')[0] + '_' + learning_algorithm.split(' ')[1]
        savename = f'{patient}_{method}_{model_name}_v2'
        learn_with_and_remember(X, y, learning_algorithm, savename, learnersaveto)

    if method.lower() == 'spectral':
        click.secho(f'Begin teaching {learning_algorithm} about patient {patient} using {method}.', fg='magenta')
        bands, band_names = get_neural_rhythm_bands()
        wait_msg = 'Extracting features... '
        online_window_size = 35 # WARNING: Do not change value without knowing what you're doing! Increasing value will not produce features for full time length. Decreasing will distort signal.
        fs = 256
        click.secho(f'Generating features for class: {class_a_name}')
        print(f'Input channels {class_a_name}: {class_a_data_hstacked.shape[0]}')
        print(f'Input length {class_a_name}: {class_a_data_hstacked.shape[1]}')
        # print('Please wait...', end='')
        # class_a_data_hstacked_padded = np.pad(class_a_data_hstacked, (class_a_data_hstacked.shape[1], class_a_data_hstacked.shape[1]), 'constant') # zero-pad both sides
        # click.secho('[Done]', fg='green')
        times, class_a_response, _, _ = neural_power_core(sig=class_a_data_hstacked, fs=fs, bands=bands, band_names=band_names, online_window_size=online_window_size, fft_window_size=2, fft_window_name='hann')
        print(f'{class_a_name} response dimensionality: {class_a_response.shape}')

        click.secho(f'Generating features for class: {class_b_name}')
        print(f'Input channels {class_b_name}: {class_b_data_hstacked.shape[0]}')
        print(f'Input length {class_b_name}: {class_b_data_hstacked.shape[1]}')
        times, class_b_response, _, _ = neural_power_core(sig=class_b_data_hstacked, fs=fs, bands=bands, band_names=band_names, online_window_size=online_window_size, fft_window_size=2, fft_window_name='hann')
        print(f'{class_b_name} response dimensionality: {class_b_response.shape}')
        print(f'{method} times dimensionality: {times.shape}')

        # Spectral response plots
        if plot_figures:
            saveto = f'./figures/{patient}/Spectral'
            class_a_savename = 'train_class_a_response'
            class_b_savename = 'train_class_b_response'
            preictal_start_time = -1
            window = fs*online_window_size
            times_in_hour = np.arange(0, times.shape[0]) / (fs/window) / 3600
            write_spectral_response_plot(times_in_hour, class_a_response, preictal_start_time, class_a_savename, saveto, x_lim_end=0.75)
            write_spectral_response_plot(times_in_hour, class_b_response, preictal_start_time, class_b_savename, saveto, x_lim_end=0.75)
        
        X, y = create_X_y(class_a_response, class_b_response)
        print('Generating pairplot, please wait...')

        # visualise ARMA feature distribution
        if plot_figures:
            savename = 'train_pairplot'
            write_spectral_pairplot(X, y, band_names, savename, saveto)

        # # train and save model
        model_name = learning_algorithm.split(' ')[0] + '_' + learning_algorithm.split(' ')[1]
        savename = f'{patient}_{method}_{model_name}_v2'
        learn_with_and_remember(X, y, learning_algorithm, savename, learnersaveto)

    click.secho(f'Completed teaching {learning_algorithm} about patient {patient} using {method}.', fg='green')

if __name__ == '__main__':
    cli()