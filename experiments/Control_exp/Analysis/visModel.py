__author__ = 'Abdullah'

import sys;
sys.path.insert(0, "../../../core");
sys.path.insert(0, "../../../tools");
sys.path.insert(0, "../../../optim");
sys.path.insert(0, "../../../datasets");
sys.path.insert(0, "../../../visualisation");




from visualiseNDMNet import *;

from deap import tools,creator, base;
import numpy as np;
import pyNDMOptim;
import ndmModel;
import random;

import activation_fn;
import output_fn;
import datasets;
from PyQt4 import QtCore, QtGui;
import pickle;


def visModel(file):
    """
    Visualises the model in the file
    :param file: file containining the pickled model
    :return: none
    """


    model = pickle.load( open( "{0}".format(file), "rb" ) );
    print model;

    # app = QtGui.QApplication(sys.argv);
    # QtCore.qsrand(QtCore.QTime(0,0,0).secsTo(QtCore.QTime.currentTime()));
    # widget2 = GraphWidget(model,'Train Set');
    # widget2.show();
    # sys.exit(app.exec_());



visModel('diabetesbest_modelFri_Oct_24_194825_2014.dat');






