#!/usr/bin/env python

import os
import click
import numpy as np
from tqdm import tqdm

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
    data_list = []
    for i, _ in enumerate(tqdm(files)):
        data = np.load(files[i])
        data_list.append(data)
        del data
    return data_list

@click.command()
@click.option('--patient', help='Patient identifier (e.g. \'chb01\')')
@click.option('--method', help='Feature extraction method. Choose either: \'ARMA\'  or \'Spectral\'')
@click.option('--learner', help='Path to trained model')
@click.option('--train', is_flag=True, help='Runs on training data if supplied. Otherwise runs on test data')
@click.option('--data', help='Root directory of train test data.')
@click.option('--saveto', help='Path to directory where generated plots will be saved.')
@click.option('--saveformat', help='File format of plot (e.g. .png, .pdf)')
def think(patient, method, learner, train, data, saveto, saveformat):
    """
    Predict seizure from EEG data in real time.
    """
    if train:
        sset = 'Train'
    else:
        sset = 'Test'
    print(f'Dataset: {sset}')
    class_a_name = 'Interictal'
    class_b_name = 'Preictal'
    class_a_files = class_files(patient, sset, class_a_name)
    class_b_files = class_files(patient, sset, class_b_name)

    print(f'Loading {len(class_a_files)} {class_a_name} arrays')
    class_a_data = load_data(class_a_files)
    print(f'Loading {len(class_a_files)} {class_b_name} arrays')
    class_b_data = load_data(class_b_files)
    
    
    


if __name__ == '__main__':
    think()