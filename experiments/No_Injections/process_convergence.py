__author__ = 'NDMProject'



import os;
import glob;
from pandas import DataFrame, read_csv;
import matplotlib.pyplot as plt;
import numpy as np;
import pickle;
"""
This processes the raw results of the experiment and produce graphs and results
"""


def process(dir):
    """
    @param: dir - is containing the files
    """

    #Check matching files
    list_files = glob.glob("./Results/"+dir+"/*_gen_fitness.csv");

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
                     y = best_costs[i], color = 'gray');

    best_cost_mean = np.array(best_costs).mean(axis = 0);
    plt.errorbar(range(len(best_cost_mean)),best_cost_mean, label = r'$Mean - no\ injections$' );
    plt.legend();
    plt.savefig(dir+'fig'+str(i)+'.png')
    plt.close();
    pickle.dump(best_cost_mean, open(dir+'.dat', 'wb'));
    

process('card');
process('heart');
process('diabetes');
process('iris');
process('cancer');
