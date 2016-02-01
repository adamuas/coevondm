__author__ = 'NDMProject'



import os;
import glob;
from pandas import DataFrame, read_csv;
import matplotlib.pyplot as plt;
import numpy as np;
import pickle;
import scipy.stats as stats;
"""
This processes the raw results of the experiment and produce graphs and results
"""


def process(dir):
    """
    @param: dir - is containing the files
    """

    #Check matching files
    list_files = glob.glob(dir+"/*_gen_fitness.csv");

    print "num files", len(list_files);
    print "list of files", list_files;

    # plt.ion();
    plt.title("${0}$".format(dir));
    plt.xlabel("$Generations(t)$");
    plt.ylabel("$Error(MSE)$");
    # plt.ylim([0.0,1.0]);
    best_costs = [];
    #read the generation fitness files
    for i, f in enumerate(list_files):
        #Read the file
        df = read_csv(f);
        #Best cost
        best_costs.append(df['best_cost'].values);
    
        #plot the graphs
        plt.errorbar(x = range(len(best_costs[i])),
                     y = best_costs[i], color = 'gray', alpha = 0.4);

        

    best_cost_mean = np.array(best_costs).mean(axis = 0);
    std_err = stats.sem(np.array(best_costs));
    std_dev = np.array(best_costs).std(axis = 0);
    plt.errorbar(x = range(len(best_cost_mean)),y = best_cost_mean, yerr =std_err,  label = r'$Mean$' );
    plt.legend();
    plt.savefig(dir+'fig_conv'+str(i)+'.png')
    plt.close();
    pickle.dump(best_cost_mean, open(dir+'.dat', 'wb'));
    
    

stats_abalone = process('abalone');
stats_echocardiogram = process('echocardiogram');
stats_hepatitis = process('hepatitis');
stats_iris = process('iris');
stats_lenses = process('lenses');
stats_parkinsons = process('parkinsons');
stats_sonar = process('sonar');
stats_vertebral2C = process('vertebral2C');
stats_vertebral3C = process('vertebral3C');
process('monks1');
process('monks2');
process('monks3');
process('ionosphere');
process('seeds');
process('bankruptcy');
process('inflamations');
process('heart');
process('card');
process('cancer');
process('diabetes');
process('spect_heart');


