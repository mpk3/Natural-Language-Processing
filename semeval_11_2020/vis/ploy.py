import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt

tests = '/home/mpk3/Natural_Language_Processing/semeval_11_2020/labeled_data/all_tests'
results  = '/*/*/*_results.txt'


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


files = glob.glob(tests+results)
result_df = build_result_df(files)
result_df.index.name = 'I'
prec_df = result_df[result_df['I'] is 'p']
result_df.groupby('I').plot.bar(x=result_df.index, y='f1-score', subplots=True)

ax = result_df[['reference','f1-score', 'test']].plot(kind='bar', subplots=True)
plt.show()
