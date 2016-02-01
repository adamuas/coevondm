__author__ = 'NDMProject'

import math;
import sys;

try:
    # import matplotlib.pyplot as plt
    import pylab as plt;
except ImportError:
    plt = False
import numpy as np;
sys.path.insert(0, "../");  #use default settings
sys.path.insert(0, "../core");
sys.path.insert(0, "../visualisation");
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../cuda");
sys.path.insert(0, "../tools");


def visualiseOutputs2D(model, dataset, name = '_'):
    """
    plots the output of the neural network model
    """
    print "Attempting to visualise";
    model_output = [];
    actual_output = dataset['OUT'];
    inputs = dataset['IN'];
    resolution = 0.01;

    #GET OUTPUTS

    X0 = np.arange(-1,1,0.025);
    X1 = np.arange(-1,1,0.025);
    numIn1Coord = len(X0);
    numIn2Coord = len(X1);
    net_outputs = np.zeros((len(X0),len(X1)), np.float32);

    for in1i in xrange(len(net_outputs)):
        for in2i in xrange(len(net_outputs[in1i])):
            invals =  [X0[in1i], X1[in2i]];

            out = model.stimulate(invals);
            net_outputs[in1i][in2i] = out[0];


    #NORMALISE
    maxOut = (net_outputs.max());
    img2D = np.zeros_like(net_outputs);
    for in1i in xrange(len(img2D)):
        for in2i in xrange(len(img2D[in1i])):
            aGrey = net_outputs[in1i][in2i]/maxOut;

            if aGrey > 1:
                aGrey = 1;
            elif aGrey <= 0:
                aGrey = 0;
            elif math.isnan(aGrey):
                aGrey = 1;

            img2D[in1i][in2i] = aGrey;

    print net_outputs;
    #im = plt.matshow(net_outputs, cmap = 'Greys');
    im2 = plt.matshow(img2D, cmap = 'Greys');

    print "OUT TEST";
    for i in xrange(len(dataset['IN'])):
        invals = dataset['IN'][i];

        print "input: ", invals;
        print "model out:", model.stimulate(dataset['IN'][i]);
        print "out_actual:", dataset['OUT'][i];



    plt.savefig(name+'.png');






