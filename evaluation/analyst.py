#!/usr/bin/env python

import os
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--patient', help='Patient identifier (e.g. \'chb01\')')
@click.option('--method', help='Feature extraction method. Choose either: \'ARMA\'  or \'Spectral\'')
@click.option('--train', is_flag=True, help='Runs on training data if supplied. Otherwise runs on test data')
@click.option('--data', help='Root directory of train test data.')
@click.option('--saveto', help='Path to directory where generated plots will be saved.')
@click.option('--saveformat', help='File format of plot (e.g. .png, .pdf)')
def think():
    """
    Predict seizure from EEG data in real time.
    """
    pass