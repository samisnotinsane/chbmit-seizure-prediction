#!/usr/bin/env python

import os
import click
from tqdm import tqdm
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@click.group()
def cli():
    pass

def class_files(patient, sset, class_name) -> list:
    root = './data' + '/' + sset + '/' + patient
    class_a = 'Interictal'
    class_b = 'Preictal'
    if class_name == class_a:
        class_a_dir = root + '/' + class_a
        return [class_a_dir + '/' + x for x in os.listdir(class_a_dir)]
    if class_name == class_b:
        class_b_dir = root + '/' + class_b
        return [class_b_dir + '/' + x for x in os.listdir(class_b_dir)]
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

def ARMA_core(sig, Ik, n_c, n, n_i, m, mp, model, wait_msg):
    k_list = []                                                # Sequential buffer for time index in prediction point _k
    a_h_k_list = []                                            # Sequential buffer for AR signal samples
    a_h_k_m_list = []
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
            a_h_k_m = np.mean(a_h, axis=0)

            # classify a_k (feature)
            p = model.predict(a_k.reshape(1, -1))

            # MA of prediction signal
            buf_MA = np.append(buf_MA, p)
            p_MA = np.mean(buf_MA)
            if len(buf_MA) == mp:
                buf_MA = np.delete(buf_MA, 0)
            
            k_list.append(_k)                                  # Add to prediction time history
            a_h_k_list.append(a_h)                             # Add to AR response history
            a_h_k_m_list.append(a_h_k_m)
            a_k_list.append(a_k)                               # Add to ARMA response history
            preds.append(p)                                    # Add to prediction history
            p_MAs.append(p_MA)                                 # Add to MA prediction history
        # END decimation condition
    # END sliding window loop
    return k_list, a_h_k_m_list, preds, p_MAs

def ARMA(sig, N, n_i, m, mp, model):
    wait_msg = 'Analysing... '
    # ARMA(X, fs, window, order, feature_mem, predict_mem)
    n_c = sig.shape[0] # channel count in EEG
    n = sig.shape[1] #  sample count in EEG
    _ks, a_h_k, p_k, p_MA_k = ARMA_core(sig, N, n_c, n, n_i, m, mp, model, wait_msg)
    times = np.hstack(_ks)
    response = np.vstack(a_h_k)
    prediction = np.hstack(p_k)
    prediction_MA = np.hstack(p_MA_k)
    return times, response, prediction, prediction_MA

def write_response_plot(times, response, preictal_start_time, savename, saveto, saveformat) -> None:
    savepath = saveto + '/' + savename + saveformat
    sns.set_palette(sns.color_palette('Set2'))
    plt.figure(figsize=(12,6))
    sns.lineplot(x=times, y=response[:,0], label='$a_1$')
    ax = sns.lineplot(x=times, y=response[:,1], label='$a_2$')
    ax.fill_between(times, 0, 1, where=times < preictal_start_time, color='#9cd34a', alpha=0.3, transform=ax.get_xaxis_transform(), label='Interictal')
    ax.fill_between(times, 0, 1, where=times > preictal_start_time, color='#ffd429', alpha=0.3, transform=ax.get_xaxis_transform(), label='Preictal')

    # plt.xticks(np.arange(0,3.76,0.25))
    # plt.xlim([0,3.75])
    plt.ylim([-1.2,2])
    plt.xlabel('Time, $h$')
    plt.ylabel('A.U.')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(savepath)
    click.secho(f'Response plot saved to: {savepath}')

def write_prediction_plot(times, prediction, MA_prediction, saveto, saveformat) -> None:
    pass

@cli.command()
@click.option('--patient', required=True, help='Patient identifier (e.g. \'chb01\')')
@click.option('--method', required=True, help='Feature extraction method. Choose either: \'ARMA\'  or \'Spectral\'')
@click.option('--learner', required=True, help='Path to trained model')
@click.option('--train', required=True, is_flag=True, help='Runs on training data if supplied. Otherwise runs on test data')
@click.option('--data', required=True, help='Root directory of train test data.')
@click.option('--saveto', required=True, help='Path to directory where generated plots will be saved.')
@click.option('--saveformat', help='File format of plot (e.g. .png, .pdf)')
@click.option('--debug', is_flag=True, help='Uses smaller portion of data for quick runs')
def think(patient, method, learner, train, data, saveto, saveformat, debug):
    """
    Predict seizure from EEG data in real time.
    """
    Path(saveto).mkdir(parents=True, exist_ok=True) # create saveto directory if not exists
    if train:
        sset = 'Train'
    else:
        sset = 'Test'
    # load data
    click.echo(f'Dataset: {sset}')
    class_a_name = 'Interictal'
    class_b_name = 'Preictal'

    class_a_files = class_files(patient, sset, class_a_name)
    class_b_files = class_files(patient, sset, class_b_name)

    if debug:
        class_a_files = class_a_files[0:1]
        class_b_files = class_b_files[0:1]

    click.echo(f'Loading {len(class_a_files)} {class_a_name} arrays')
    class_a_data = load_data(class_a_files)
    click.echo(f'Loading {len(class_a_files)} {class_b_name} arrays')
    class_b_data = load_data(class_b_files)

    X = merge_data(class_a_data, class_b_data)
    click.secho(f'Loaded merged data: {X.shape}')

    click.secho(learner, fg='yellow')
    model = load(learner)
    print(f'Model: {model}')
    click.secho(f'Hyperparameters:\n {model.get_params()}')

    if method == 'ARMA':
        # initialise ARMA parameters
        fs = 256           # sampling frequency in Hz
        window = 512       # window width
        order = 2          # Second order model AR(2)
        feature_mem = 30   # MA smoothing for AR features
        predict_mem = 5    # MA smoothing for prediction signal
        
        print(f'Input length: {X.shape[1]}')
        print(f'Input channels: {X.shape[0]}')
        print(f'ARMA{order, feature_mem}')
        print(f'Window size: {window}')
        print(f'Sampling frequency: {fs} Hz')
        print(f'Prediction frequency: {fs/window} Hz')
        print(f'Prediction smoothing: {predict_mem}')

        # online prediction
        times, response, prediction, MA_prediction = ARMA(X, window, order, feature_mem, predict_mem, model)

        print('times:', times.shape)
        print('response:', response.shape)

        
        class_a_data_hstacked = np.hstack(class_a_data)
        times_in_hour = np.arange(0, times.shape[0]) / (fs/window) / 3600

        
        preictal_start_time = np.rint(np.max( (np.arange(0, class_a_data_hstacked.shape[1]) / fs) )) / 3600
        print('preictal_start_time:', preictal_start_time)

        # plots
        savename = 'ARMA_response'
        write_response_plot(times_in_hour, response, preictal_start_time, savename, saveto, saveformat)
        # write_prediction_plot(times, prediction, MA_prediction, saveto, saveformat)

if __name__ == '__main__':
    cli()