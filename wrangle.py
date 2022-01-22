# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

import wfdb  # PyPi package for loading ecg and annotations
from sklearn.model_selection import train_test_split


#################LOAD FILE FUNCTION##############

def load_ecg(file):
    # load the ecg
    # example file: 'mit-bih-arrhythmia-database-1.0.0/101'

    # load the ecg
    record = wfdb.rdrecord(file)
    # load the annotation
    annotation = wfdb.rdann(file, 'atr')

    # extract the signal
    p_signal = record.p_signal

    # verify frequency is 360
    assert record.fs == 360, 'sample freq is not 360'

    # extract symbols and annotation index
    atr_sym = annotation.symbol
    atr_sample = annotation.sample

    return p_signal, atr_sym, atr_sample


################### MAKE DATASET FUNCTION ##################

def make_dataset(pts, num_sec, fs, abnormal):
    '''
    function for making dataset ignoring non-beats

    input:
        pts - list of patients
        num_sec = number of seconds to include before and after the beat
        fs = frequency

    output: 
        X_all = signal (nbeats , num_sec * fs columns)
        Y_all = binary is abnormal (nbeats, 1)
        sym_all = beat annotation symbol (nbeats,1)
    '''

    # initialize numpy arrays
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1, num_cols))
    Y_all = np.zeros((1, 1))
    sym_all = []

    # list to keep track of number of beats across patients
    max_rows = []

    for pt in pts:
        file = data_path + pt

        p_signal, atr_sym, atr_sample = load_ecg(file)

        # grab the first signal
        p_signal = p_signal[:, 0]

        # make df to exclude the nonbeats
        df_ann = pd.DataFrame({'atr_sym': atr_sym,
                              'atr_sample': atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal + ['N'])]

        X, Y, sym = build_XY(p_signal, df_ann, num_cols, abnormal)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all, X, axis=0)
        Y_all = np.append(Y_all, Y, axis=0)
    # drop the first zero row
    X_all = X_all[1:, :]
    Y_all = Y_all[1:, :]

    # check sizes make sense
    assert np.sum(
        max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert Y_all.shape[0] == X_all.shape[0], 'number of X, Y rows messed up'
    assert Y_all.shape[0] == len(sym_all), 'number of Y, sym rows messed up'

    return X_all, Y_all, sym_all

################ XY SPLIT FUNCTION  #######################


def build_XY(p_signal, df_ann, num_cols, abnormal):
    '''
    this function builds the X,Y matrices for each beat
     it also returns the original symbols for Y
    '''

    num_rows = len(df_ann)

    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows, 1))
    sym = []

    # keep track of rows
    max_row = 0

    for atr_sample, atr_sym in zip(df_ann.atr_sample.values, df_ann.atr_sym.values):

        left = max([0, (atr_sample - num_sec*fs)])
        right = min([len(p_signal), (atr_sample + num_sec*fs)])
        x = p_signal[left: right]
        if len(x) == num_cols:
            X[max_row, :] = x
            Y[max_row, :] = int(atr_sym in abnormal)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row, :]
    Y = Y[:max_row, :]
    return X, Y, sym

##################### WRANGLE FUNCTION ############


def wrangle():
    # Define path where data reside
    data_path = '/Users/jaredgodar/codeup-data-science/ecg_anomaly_detection/physionet.org/files/mitdb/1.0.0/'

    # list of patients
    pts = ['100', '101', '102', '103', '104', '105', '106', '107',
           '108', '109', '111', '112', '113', '114', '115', '116',
           '117', '118', '119', '121', '122', '123', '124', '200',
           '201', '202', '203', '205', '207', '208', '209', '210',
           '212', '213', '214', '215', '217', '219', '220', '221',
           '222', '223', '228', '230', '231', '232', '233', '234']

    # list of nonbeat and abnormal annotations
    nonbeat = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`',
               '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@', 'Q', '?']
    abnormal = ['L', 'R', 'V', '/', 'A', 'f',
                'F', 'j', 'a', 'E', 'J', 'e', 'S']
    # Load annotations and look at distribution of heartbeat types
    print('Define path and patient numbers...')
    df = pd.DataFrame()

    for pt in pts:
        file = data_path + pt
        annotation = wfdb.rdann(file, 'atr')
        sym = annotation.symbol

        values, counts = np.unique(sym, return_counts=True)
        df_sub = pd.DataFrame(
            {'sym': values, 'val': counts, 'pt': [pt]*len(counts)})
        df = pd.concat([df, df_sub], axis=0)
    print('Annotations loaded...')
    # break into normal, abnormal or nonbeat
    df['cat'] = -1
    df.loc[df.sym == 'N', 'cat'] = 0
    df.loc[df.sym.isin(abnormal), 'cat'] = 1

    # Files are named by patient number, already listed in the pts array, in the path defined
    file = data_path + pts[0]

    p_signal, atr_sym, atr_sample = load_ecg(file)
    print('Loading ecg files...')
    # Get index of any abnormal beats
    ab_index = [b for a, b in zip(atr_sym, atr_sample) if a in abnormal][:10]

    # define parameters
    num_sec = 3
    fs = 360
    print('Creating dataset...')
    print('(This usually takes a while...)')
    X_all, Y_all, sym_all = make_dataset(pts, num_sec, fs, abnormal)

    # Create dataframes
    X_all_df = pd.DataFrame(X_all)
    Y_all_df = pd.DataFrame(Y_all)
    sym_all_df = pd.DataFrame(sym_all)
    print('Dataframes created...')
