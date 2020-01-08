import re
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Matplot settings
pd.options.display.mpl_style = 'default'

# Repos
TESTS = '/home/mpk3/Natural_Language_Processing/' +\
    'semeval_11_2020/labeled_data/all_tests'
RESULTS = TESTS + '/*/*/*_results.txt'
FEATURES = TESTS + '/*/*/*_features.txt'
TRAINING = TESTS + '/*/*/*_iterations.txt'


def build_feature_dictionary(files):
    '''Gets the feature sets for each test '''
    for file_in in files:
        ext = file_in.split('/')
        test_number = ext[7]
        optimization = ext[8]
        reference_number = ext[-1][0]
        text_wrapper = open(file_in, 'r')

def build_result_df(files):
    '''Builds dataframe of the result files'''
    df_list = []
    for file_in in files:
        ext = file_in.split('/')
        test_number = ext[7]
        optimization = ext[8]
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
    
    for test in results:
        for ind, obj in test.groupby('reference'):
            t_str = 'Test Set: ' + str(ind)
            y_list = ['f1-score', 'precision', 'recall']
            obj.plot.bar(x='opt', y=y_list, title=t_str)
    plt.show()


def build_iteration_df(files):
    '''Creates DataFrame for training information'''
    df = pd.DataFrame(columns=['Iteration', 'Time', 'Loss',
                               'Test', 'Opt', 'Reference'])
    for f_in in files:
        ext = f_in.split('/')
        test_number = ext[7]
        optimization = ext[8]
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


# Driver Code
tfiles = glob.glob(TRAINING)
rfiles = glob.glob(RESULTS)
# ffiles = glob.glob(FEATURES)


# TRAINING
dft = build_iteration_df(tfiles)

t1_df_lbfgs = dft[dft['Opt'] == 'lbfgs']
t2_df_ap = dft[dft['Opt'] == 'ap']
t3_df_l2sgd = dft[dft['Opt'] == 'l2sgd']
t3_df_pa = dft[dft['Opt'] == 'pa']
t3_df_arow = dft[dft['Opt'] == 'arow']

all_iterations = [t1_df_lbfgs, t2_df_ap,
                  t3_df_l2sgd, t3_df_pa, t3_df_arow]
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

objs = []
axs = [ax0, ax1, ax2]
for opt in all_iterations:
    for ind, obj in opt.groupby('Test'):
        t_str = obj.iloc[0]['Test']
        fig, ax = plt.subplots()
        for jind, jobj in obj.groupby('Reference'):
            r_str = jobj.iloc[0]['Reference']
            ax = jobj.plot(ax=ax, x='Iteration', y='Loss', loglog=True,
                           linestyle='dotted', linewidth=1.3,
                           title='Test: ' + t_str + 'Ref: ' + str(r_str))

plt.show()


for ind, obj in all_iterations[0].groupby('Test'):
    print(ind)
    ax = obj.plot(ax=ax, kind='line',
                  x='Iteration', y='Loss')
    objs.append(obj)

for ind, obj in objs[0].groupby('Test'):
    ax = obj.plot(ax=ax, kind='line',
                  x='Iteration', y='Loss')

plt.show()

t1_df_tl = t1_df_t.drop('Time', axis=1)
t1_df_tl.groupby(['Reference']).groupby()

.plot(x='Iteration', y='Loss')
plt.show()



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


t1_dfr.plot.bar(x='reference', y='f1-score', subplots=True)
plt.show(subplots=True)








####### Stuff for Feature files not needed right now

tests = {}

file_in = ffiles[0]
ext = file_in.split('/')
test_number = ext[7]
optimization = ext[8]
reference_number = ext[-1][0]
text_wrapper = open(file_in, 'r')
lines = text_wrapper.readlines()
features = set(eval(lines[0][9:].strip()))
keys = tests[test_number].keys()
#if test_number in keys:




