__author__ = 'Abdullah'

import sys;
sys.path.insert(0, "../../../../core");
sys.path.insert(0, "../../../../optim");
sys.path.insert(0, "../../../../visualisation");
sys.path.insert(0, "../../../../datasets");
sys.path.insert(0, "../../../../tools");

import pickle;
import glob;

from visualiseNDMNet import *;
import ndmCoevoOptim;
from deap import tools,creator, base;
import numpy as np;
import pyNDMOptim;
import ndmModel;
import random;
import activation_fn;
import output_fn;

from PyQt4 import QtCore, QtGui;


def visualize(params, folder):

    #Check matching files
    list_files = glob.glob(folder+"/*_model*.dat");
    print list_files;
    print "No. of files:", len(list_files);
    ### LOAD All the Models ###
    for model_file in list_files:
        
        model = pickle.load(open(model_file,'rb'));
        
        print ">Reading model :",model_file;
       
        print ">creating model...";
        print model;
        ndmmodel = ndmModel.ndmModel(params['numI'], params['numH'], params['numO'], model['best_sol']);

        print ">model created!", type(ndmmodel);

    
        #visualise the model
        print ">visualizing model";
        app = QtGui.QApplication(sys.argv);
        QtCore.qsrand(QtCore.QTime(0,0,0).secsTo(QtCore.QTime.currentTime()));
        widget2 = GraphWidget(ndmmodel,'Test Set');
        widget2.show();

        
params = dict();
params['numI'] = 4;
params['numH'] = 2;
params['numO'] = 1;
visualize(params,'iris');

