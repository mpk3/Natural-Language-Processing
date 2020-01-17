import re
# import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# Repos
TESTS = '../labeled_data/all_tests'
RESULTS = TESTS + '/*/*/*_results.txt'
FEATURES = TESTS + '/*/*/*_features.txt'
TRAINING = TESTS + '/*/*/*_iterations.txt'


def build_feature_dictionary(files):
    '''Gets the feature sets for each test '''
    for file_in in files:
        ext = file_in.split('/')
        test_number = ext[3]
        optimization = ext[4]
        reference_number = ext[-1][0]
        text_wrapper = open(file_in, 'r')

def build_result_df(files):
    '''Builds dataframe of the result files'''
    df_list = []
    for file_in in files:
        ext = file_in.split('/')
        test_number = ext[3]
        optimization = ext[4]
        reference_number = ext[-1][0]
        text_wrapper = open(file_in, 'r')
        text_list = text_wrapper.readlines()
        split_list = [s.split() for s in text_list]
        frame = list(filter(None, split_list))
        columns = frame.pop(0)
        index = [l.pop(0) for l in frame]
        frame[4].remove('avg')
        frame[3].remove('avg')
        frame[2].remove('avg')

        df = pd.DataFrame(data=frame, index=index, columns=columns)
        df['opt'] = optimization
        df['reference'] = int(reference_number)
        df['test'] = test_number
        df['recall'] = df['recall'].astype('float')
        df['precision'] = df['precision'].astype('float')
        df['f1-score'] = df['f1-score'].astype('float')
        df['support'] = df['support'].astype('int32')
        df_list.append(df)
    end_frame = pd.concat(df_list, axis=0)
    return end_frame


def plot_results(results):
    '''Plots graphs for the results of each test set'''
    fig, ax = plt.subplots(3, 3)
    ncol = 0
    for test in results:
        nrow = 0
        for ind, obj in test.groupby('reference'):
            y_list = ['f1-score', 'precision', 'recall']
            ax1 = obj.plot.bar(x='opt', y=y_list, ax=ax[nrow, ncol])
            ax1.set_xlabel('')
            ax1.tick_params(axis='x',labelrotation=.25)
            nrow = nrow + 1
        ncol = ncol + 1

    ax[0, 0].\
        set_title('Testing Phase 1: No Sentiment', fontsize='x-small')
    ax[0, 1].\
        set_title('Testing Phase 2: Token Level Sentiment', fontsize='x-small')
    ax[0, 2].\
        set_title('Testing Phase 3: n-level Sentiment', fontsize='x-small')
    ax[0, 0].set_ylabel('Unigram')
    a1t = ax[0, 2].twinx()
    a1t.set_ylabel('n sentiment')
    ax[1, 0].set_ylabel('Trigram')
    a2t = ax[1, 2].twinx()
    a2t.set_ylabel('n-1, n+1 sentiment')
    a3t = ax[2, 2].twinx()
    a3t.set_ylabel('n-2, n+2 sentiment')
    ax[2, 0].set_ylabel('5-gram')
    fig.suptitle('Test Results by Test Phase')
    plt.subplots_adjust(bottom=.05, top=.9)
    plt.show()


def build_iteration_df(files):
    '''Creates DataFrame for training information'''
    df = pd.DataFrame(columns=['Iteration', 'Time', 'Loss',
                               'Test', 'Opt', 'Reference'])
    for f_in in files:
        ext = f_in.split('/')
        test_number = ext[3]
        optimization = ext[4]
        reference_number = ext[-1][0]
        with open(f_in, 'r') as file_in:
            line = file_in.readline()
            while line:
                it = re.match('Iter.*', line)
                if it is not None:
                    data_map = {}
                    it = it.group(0).split()
                    data_map['Iteration'] = int(it[1])
                    data_map['Time'] = float(it[2].split('=')[1])
                    data_map['Loss'] = float(it[3].split('=')[1])
                    data_map['Test'] = test_number
                    data_map['Opt'] = optimization
                    data_map['Reference'] = int(reference_number)
                    df = df.append(data_map, ignore_index=True)
                line = file_in.readline()
    return df


def plot_training(training_dataframe):
    '''Creates Training Graph'''
    fig, ax = plt.subplots(5, 3)
    col = 0
    s = ['Unigram', 'Trigram', '5-Gram']
    s2 = ['n', 'n+1, n-1', 'n+2, n-2']
    
    for j, test in training_dataframe.groupby('Test'):
        row = 0
        for ind, obj in test.groupby('Opt'):
            # print(ind)
            for jind, jobj in obj.groupby('Reference'):
                ax1 = jobj.plot(x='Iteration', y='Loss',
                                ax=ax[row, col], loglog=True)
                ax1.set_xlabel('Optimization Iteration - Log Scale', fontsize='x-small')
                ax1.set_yticklabels([], minor=True)
                ax1.set_yticklabels([])
                ax1.set_yticks([], minor=True)
                ax1.set_yticks([])
                ax1.set_xticklabels([], minor=True)
                ax1.set_xticklabels([])
                ax1.set_xticks([], minor=True)
                ax1.set_xticks([])
                if col<=1:
                    ax1.legend(s)
                else:
                    ax1.legend(s2)
            row = row + 1
        col = col + 1
    ax[0,0].\
        set_title('Testing Phase 1: No Sentiment', fontsize='x-small')
    ax[0,1].\
        set_title('Testing Phase 2: Token Level Sentiment', fontsize='x-small')
    ax[0,2].\
        set_title('Testing Phase 3: n-level Sentiment', fontsize='x-small')
    ax[0,0].\
        set_ylabel('ap')
    ax[1,0].set_ylabel('arow')
    ax[2,0].set_ylabel('l2sgd')
    ax[3,0].set_ylabel('lbfgs')
    ax[4,0].set_ylabel('pa')
    fig.suptitle('Training Loss over Iterations by Test Phase and Algorithm (Log-Log Scale)')
    plt.subplots_adjust(bottom=.05, top=.9)
    plt.show()


# Driver Code
tfiles = glob.glob(TRAINING)
rfiles = glob.glob(RESULTS)
# ffiles = glob.glob(FEATURES)


# TRAINING
dft = build_iteration_df(tfiles)
plot_training(dft)

# RESULTS
# File I/O and Pandas DataFrame
dfr = build_result_df(rfiles)

# Trim DataFrames
t1_dfr = dfr[dfr['test'] == 'test1'].drop('macro').drop('micro').\
    drop('o').drop('weighted').drop('support', axis=1)
# a = [(ind, obj) for ind, obj in t1_dfr.groupby('reference')]
t2_dfr = dfr[dfr['test'] == 'test2'].drop('macro').drop('micro').\
    drop('o').drop('weighted').drop('support', axis=1)
t3_dfr = dfr[dfr['test'] == 'test3'].drop('macro').drop('micro').\
    drop('o').drop('weighted').drop('support', axis=1)

all_results = [t1_dfr, t2_dfr, t3_dfr]
plot_results(all_results)
