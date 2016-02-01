__author__ = 'Abdullah'

import sys;
sys.path.insert(0, "../optim");
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../core");
sys.path.insert(0, "../visualisation");

import ndmCoevoOptim;
import ndmCoevoBase;
import ndmModel;
import numpy as np;
import datasets;
from benchmark import proben1_bechmark as proben1;
from benchmark import lab_bencmark;
import kfold;
import profile;
import visualisation.visualiseOutputs2D as vis2d;
from PyQt4 import QtCore, QtGui
from visualiseNDMNet import *;

coevo = ndmCoevoOptim.ndmCoevoOptim();
errors_train =[];
errors_test = [];

benchmark = proben1();
lab_bencmark = lab_bencmark();
K  = 10;

# D = kfold.kfold(D = benchmark.mushroom()['train'],numFolds = K);
D2 = kfold.kfold(D = lab_bencmark.iris()['train'],numFolds = K);
for i in xrange(1):

    print ">>>", i;

    coevo.init_populations();
    # coevo.train_set = D2[i][0];
    # coevo.validation_set = D2[i][1];
    profile.run("coevo.coevolve()");









