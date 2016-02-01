__author__ = 'NDMProject'

import os;
import glob;
from pandas import DataFrame, read_csv;
import matplotlib.pyplot as plt;
import numpy as np;

"""
This processes the raw results of the experiment and produce graphs and results
"""


def process(dir):
    """
    @param: dir - is containing the files
    """

    #Check matching files
    list_files = glob.glob("./Results/"+dir+"/*_Performance.csv");

    print "num files", len(list_files);
    print "list of files", list_files;


    #Process the files
    files_data = [];
    train_errs = np.zeros((len(list_files)));
    test_errs = np.zeros((len(list_files)));
    # Import the Files
    for i,f in enumerate(list_files):
        #read files into data frames
        files_data.append(read_csv(f));
        train_errs[i] = files_data[i]['train_err'][0];
        test_errs[i] = files_data[i]['test_err'][0];

    #- get vital statistical information
    stats = dict();
    stats['avg_train_err'] = train_errs.mean();
    stats['std_train_err ']= train_errs.std();
    stats['avg_test_err'] = test_errs.mean();
    stats['std_test_err'] = test_errs.std();
    stats['conv_train_err'] = 0;
    stats['conv_test_err'] = 0;

    count_tr_e = 0;
    count_ts_e = 0;

    for tr_e,ts_e in zip(train_errs,test_errs):
        if tr_e < stats['avg_train_err']:
            count_tr_e += 1;
        if ts_e < stats['avg_test_err']:
            count_ts_e += 1;

    stats['conv_train_err'] = count_tr_e/float(len(train_errs));
    stats['conv_test_err'] = count_ts_e/float(len(test_errs));


    #Save
    stats_df = DataFrame(stats,index = range(1));
    stats_df.to_csv(dir+'.csv');
    
    #-plot the combined graphs of the files
    #plot
    plt.title(dir);
    plt.ylabel('Error (MSE)');
    plt.bar(range(2),[stats['avg_train_err'],stats['avg_test_err']] );

    plt.savefig(dir+'.png');

    return stats;




def combined_plot(DIRS):
    """
    plots data from a set of directorties
    """
    
    stats_aussie_cc = process('card');
    stats_cancer = process('cancer');
    stats_diabetes = process('diabetes');
    stats_heart = process('heart');
    stats_iris = process('iris');
    stats_xor = process('XOR');

    



stats_aussie_cc = process('card');
stats_cancer = process('cancer');
stats_diabetes = process('diabetes');
stats_heart = process('heart');
stats_iris = process('iris');




