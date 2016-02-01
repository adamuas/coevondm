"""
A set of functions for analysing to help analyse the strategy used by a neural network
@author: Abdullahi shuiabu
@author-email: abdullah.adam89@hotmail.com
#Kindly send suggestions and corrections to this email.
"""

import os, sys, inspect
sys.path.insert(0,"../optim");
sys.path.insert(0,"../");
sys.path.insert(0,"../core");
sys.path.insert(0,"../datasets");
sys.path.insert(0,"../visualisation");

import numpy;
import ndm;
import datasets;
import commons;
from optimEnDec import *;
import dictWriter;
import csvReader;
import preprocessing;
import matplotlib.mathtext as mathtext
import numpy as np;
import networkx as nx;

import csv;

#import visualiseNet
from visualiseNet import *;
from PyQt4 import QtCore, QtGui

#plot lib
import matplotlib.pyplot as plt;

import netParams;
#import optimParams;


debug = False;
debug_A = False;

def connectionDensity(net):
    """ Measures the ratio of connections made to all transfer function
    to the ratio of the possible connections possible"""


    #weight and node functions
    weightFns = netParams.weightFns;    
    nodeFns = netParams.nodeFns;

    #create transfer function combination
    Comb = [(w,n) for w in weightFns for n in nodeFns];
    CombLabels =[str(x) for x in Comb];
    numComb = len(Comb);
    connDensity = numpy.zeros((numComb),numpy.float32);
    
    if debug_A:
        print "* Possible Combinations: ", Comb;
        print "* CoExistenceMat:", connDensity;
        print "* Weight Functions: ", weightFns;
        print "* Node Functions: ", nodeFns;

    #find the transfer functions that coexist in the given network model
    for node in net.nodes:

        if isinstance(node,inactive_node) is True:
            if debug_A:
                print "inactive node encountered, skipping.";
            continue;

        
        #get weight function
        weightFn = node.weightFn;
        nodeFn= node.nodeFn;

        #skip input nodes
        if weightFn == constants.NONE:
            if debug_A:
                print "* Skipping input node.";
            continue;

        #form transfer function
        transFn = (weightFn, nodeFn);
        
        
            
        
        #get the connectivity activity 
        connActivenessMat = net.conn_active[:][node.map_id];
        density = sum(connActivenessMat)/float(len(connActivenessMat));
        
        
        #debug
        if debug_A:
            print "transFn:", transFn;
            print "Connection Activeness Matrix:", connActivenessMat;
            print "Connection density:", density;

        #get index of the tranfer function
        transFn_i = Comb.index(transFn);
        connDensity[transFn_i] = density;

    #return the connection densities
    return connDensity;

def coexistenceLikelihood(net, filename = None):
    """
    Measures the likelihood of two transfer functions to be in the same network (i.e. best network)

    @params: network
    @returns: Coexistence likelihood matrix as Numpy Matrix.
    
    """

    #weight and node functions
    weightFns = netParams.weightFns;
    nodeFns = netParams.nodeFns;

    #create transfer function combination
    Comb = [(w,n) for w in weightFns for n in nodeFns];
    CombLabels =[str(x) for x in Comb];
    numComb = len(Comb);
    CoexistModelMat = numpy.zeros((numComb,numComb),numpy.float32);
    CoexistPathMat = numpy.zeros((numComb,numComb),numpy.float32);
    CoexistLayerMat = numpy.zeros((numComb,numComb),numpy.float32);
    
    if debug:
        print "* Possible Combinations: ", Comb;
        print "* CoExistenceMat(Model):", CoexistModelMat;
        print "* CoExistenceMat(Path):", CoexistPathMat;
        print "* CoExistenceMat(Layer):", CoexistLayerMat;
        print "* Weight Functions: ", weightFns;
        print "* Node Functions: ", nodeFns;


    #find the transfer functions that coexist in the given network model
    for node_j in net.nodes:

        if isinstance(node_j,inactive_node) is True:
            if debug:
                print "inactive node encountered, skipping.";
            continue;

        
        #get weight function
        weightFn_j = node_j.weightFn;
        nodeFn_j = node_j.nodeFn;

        #skip input nodes
        if weightFn_j == constants.NONE:
            if debug:
                print "* Skipping input node.";
            continue;

        #form transfer function
        transFn_j = (weightFn_j, nodeFn_j);
        transFn_i = None;

        if debug:
            print "transFn_j", transFn_j;
            
        
        #get all incoming connections from other nodes, (if any)
        for node_i in net.nodes:
            
            #is an active node and note same node
            if isinstance(node_i, inactive_node) is not True and node_i.node_id != node_j.node_id:
                #get weight function and node function
                weightFn_i = node_i.weightFn;
                nodeFn_i = node_i.nodeFn;
                
                #skip input nodes
                if weightFn_i == constants.NONE:
                    if debug:
                        print "* Skipping input node.";
                    continue;
                
                #form transfer function
                transFn_i = (weightFn_i, nodeFn_i);

                if debug:
                    print "transFn_i", transFn_i;
                    
                #get transfer functions indices and mark coordinates on coexistence matrix
                itransFn_i = Comb.index(transFn_i);
                itransFn_j = Comb.index(transFn_j);

                ##############################(1) COEXIST IN SAME NETWORK #################
                CoexistModelMat[itransFn_i][itransFn_j] += (1);
                    
                ##############################(2) COEXIST ON SAME PATH ###################
                # Are connected?
                if net.conn_active[node_i.map_id][node_j.map_id] == constants.ACTIVE:

                    

                    #get connection weight
                    weight_ij = net.connMatrix[node_i.map_id][node_j.map_id];
                     

                    #skip input nodes
                    if weightFn_i == constants.NONE:
                        if debug:
                            print "* Skipping input node.";
                        continue;


                    #form transfer function
                    transFn_i = (weightFn_i, nodeFn_i);

                    if debug:
                        print "transFn_i", transFn_i;

                    #get transfer functions indices and mark coordinates on coexistence matrix
                    itransFn_i = Comb.index(transFn_i);
                    itransFn_j = Comb.index(transFn_j);

                    CoexistPathMat[itransFn_i][itransFn_j] += (1);
                    
                ################################(3) COEXIST IN SAME LAYER #######################
                if node_i.type == node_j.type:
                    CoexistLayerMat[itransFn_i][itransFn_j] += (1); 
                    
                    
    
    return {'onPath': CoexistPathMat,'inLayer': CoexistLayerMat, 'inModel': CoexistModelMat };    


def coexistenceConnStrength(net):
    """
    Measures the relative strength of connections between coexisting transfer functions in a neural network model

    @params: network
    @returns: Coexistence strenth 
    
    """

    #weight and node functions
    weightFns = netParams.weightFns;    
    nodeFns = netParams.nodeFns;

    #create transfer function combination
    Comb = [(w,n) for w in weightFns for n in nodeFns];
    CombLabels =[str(x) for x in Comb];
    numComb = len(Comb);
    CoexistMat = numpy.zeros((numComb,numComb),numpy.float32);
    
    if debug:
        print "* Possible Combinations: ", Comb;
        print "* CoExistenceMat:", CoexistMat;
        print "* Weight Functions: ", weightFns;
        print "* Node Functions: ", nodeFns;

    #find the transfer functions that coexist in the given network model
    for node_j in net.nodes:

        if isinstance(node_j,inactive_node) is True:
            if debug:
                print "inactive node encountered, skipping.";
            continue;

        
        #get weight function
        weightFn_j = node_j.weightFn;
        nodeFn_j = node_j.nodeFn;

        #skip input nodes
        if weightFn_j == constants.NONE:
            if debug:
                print "* Skipping input node.";
            continue;

        #form transfer function
        transFn_j = (weightFn_j, nodeFn_j);
        transFn_i = None;

        if debug:
            print "transFn_j", transFn_j;
            
        
        #get all incoming connections from other nodes, (if any)
        for node_i in net.nodes:
            
            cumulative_conn_weight = sum(net.connMatrix[:][node_j.map_id]);
            
            #debug
            if debug:
                print "cumulative connection weight :", cumulative_conn_weight;
            
            if isinstance(node_i, inactive_node) is not True and node_i.node_id != node_j.node_id:
                if net.conn_active[node_i.map_id][node_j.map_id] == constants.ACTIVE:

                    #get weight function and node function
                    weightFn_i = node_i.weightFn;
                    nodeFn_i = node_i.nodeFn;

                    #get connection weight
                    weight_ij = abs(net.connMatrix[node_i.map_id][node_j.map_id]);
                    if debug:
                        print "* weight_ij:", weight_ij;

                    #skip input nodes
                    if weightFn_i == constants.NONE:
                        if debug:
                            print "* Skipping input node.";
                        continue;


                    #form transfer function
                    transFn_i = (weightFn_i, nodeFn_i);

                    if debug:
                        print "transFn_i", transFn_i;

                    #get transfer functions indices and mark coordinates on coexistence matrix
                    itransFn_i = Comb.index(transFn_i);
                    itransFn_j = Comb.index(transFn_j);

                    CoexistMat[itransFn_i][itransFn_j] += weight_ij;
                    
    
    return CoexistMat;

def pathLikelihood(CoexistMat):
    """ This is a higher order functions which takes in the inputs of the coexistence likelihood to form the most likely paths"""

    #create graph
    G = nx.Graph();
    
    for tfn_i in xrange(len(CoexistMat)):
        for tfn_j in xrange(len(CoexistMat)):
            
            
            path = (tfn_i,tfn_j);
            

def coactivationLikelihood(net, filename = None):
    """ Measures the likelihood of two transfer functions to activate for the same pattern

    @params: network
    @returns: Co-activation likelihood matrix as Numpy Matrix
    """


    pass;


def clusterCoexistenceMatrix(CLM, filename = None):
    """ Performs clustering on a coexistence matrix, then visualises the clusters and
    saves as a file

    @params: coexistence likelihood matrix, saveAs filename
    @return: None
    """
    pass;


def getActivationSequence():
    """ Performs analysis on a co-activation likelihood matrix and returns the sequence of activations """
    pass;


def getMostlikelySequence():
    """ Returns the most likely sequence of activation in from the coactivation likelihood Matrix"""

    pass;

#debug
debug_matavg = True;
def genCoexistMatAvg(STAT_filename, CombLabels = None, saveAs = None, show = True):
    """Retrieves readings from a stat file , calculates the average and then generate the heatmap """
    
    stat_f = open(STAT_filename, 'r');
    reader = csv.reader(stat_f, delimiter = '\t', quoting = csv.QUOTE_NONNUMERIC);
    
    #average matrix
    coexistMatAvg = [];
    
    #Row index of coexist matrix
    COEXIST_MAT_COL = 4;
    
    
    row_count = 0;
    for row in reader:
       
        print row;
        
        
        #increament
        row_count += 1;
        
        if row_count  == 2:
            break;
        
    #print type(coexistMatAvg);
    #print coexistMatAvg;
    
    #get average
    coexistMatAvg = coexistMatAvg *  float(1.0/row_count);
    
        
    #visualise and save
    visualise_coexistence_likelihood(coexistMatAvg,combLabels,saveAs,show);
        
        
        
        
    
    
    
    


#################################################################
#################################################################
#################################################################
#### VISUALISATION BIT ##########################################
#################################################################
#################################################################
#################################################################

def visualise_transfer_fn_likelihood(TFnsLikelihood, xlabels = None, ylabels = None, saveAs = None, show = True):
    """
    Visualises the Transfer functions likelihood
    """

    plt.matshow(TFnsLikelihood,  cmap = plt.cm.gray_r);
    xlabels = [str(x) for x in xlabels];
    ylabels = [str(y) for y in ylabels];
    
    if xlabels != None and ylabels != None:
        plt.xticks(range(len(xlabels)),xlabels, rotation = 'vertical');
        plt.yticks(range(len(ylabels)), ylabels);
        
    plt.ylabel(r'Output function ($f(g(.))$');
    plt.xlabel(r'Activation function ($g(.))$');
    plt.autoscale();
    
    if saveAs != None:
        plt.savefig(saveAs);
        
    #if show:    
    #    plt.show();
    
    
def visualise_coexist_conn_strength (relWeightConnMat, CombLabels = None, saveAs = None, show = True):
    """ Visualises the relative weights between connections on average
        
        @params:
        @output:
    
    """
    
    numComb = len(CombLabels);
    plt.matshow(relWeightConnMat,  cmap = plt.cm.OrRd);
    plt.xticks(range(numComb),CombLabels, rotation = 'vertical');
    plt.yticks(range(numComb), CombLabels);
    plt.ylabel(r'Transfer function ($tf_j)$');
    plt.xlabel(r'Transfer function ($tf_i)$');
    plt.autoscale();
    
    if saveAs != None:
        plt.savefig(saveAs);
        
    #if show:    
    #    plt.show();
    #
    
    
def visualise_coexistence_likelihood(coexistMat, CombLabels = None, saveAs = None, show = True):
    """ Visualises the likelihood of two transfer functions coexisting on thesame connection path
        
        @params: it takes in a coexistence matrix and plots the matrix in 2D
        @output:
    
    """
    
    numComb = len(CombLabels);
    
    plt.matshow(coexistMat,  cmap = plt.cm.Blues);
    plt.xticks(range(numComb),CombLabels, rotation = 'vertical');
    plt.yticks(range(numComb), CombLabels)
    plt.ylabel(r'Transfer function ($tf_j)$');
    plt.xlabel(r'Transfer function ($tf_i)$');
    plt.autoscale();
    
    if saveAs != None:
        plt.savefig(saveAs);
        
    if show:
        plt.show();
    
def visalise_conn_density(connDensityMat, labels = None, saveAs = None, show = True):
    """
    Visualises the density of connections to transfer functions.
    
    @params: 
    """
    fig = plt.figure();
    fig = plt.barh(range(len(connDensityMat)), connDensityMat, alpha = 0.5);
    plt.yticks(range(len(connDensityMat)), labels);
    plt.ylabel(r'Transfer function $(g(.), f(g(.)))$');
    plt.xlabel(r'Connection Density');
    plt.title('Transfer functions connection density');

    if saveAs != None:
        plt.savefig(saveAs);
        
    if show :
        plt.show();
    

    
def hinton_weight(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2));

    ax.patch.set_facecolor('gray');
    ax.set_aspect('equal', 'box');
    ax.xaxis.set_major_locator(plt.NullLocator());
    ax.yaxis.set_major_locator(plt.NullLocator());

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view();
    ax.invert_yaxis();
    
    #if show:
    #    plt.show();
    


##TEST COEXISTLIKELIHOOD##'
# Clustering: http://docs.scipy.org/doc/scipy/reference/cluster.vq.html#module-scipy.cluster.vq
#weight and node functions
##weightFns = netParams.weightFns;    
##nodeFns = netParams.nodeFns;
##
##D = np.random.rand(4,4);
##
##visualise_transfer_fn_likelihood(D, xlabels = weightFns, ylabels = nodeFns);
##
###create transfer function combination
##Comb = [(w,n) for w in weightFns for n in nodeFns];
##CombLabels =[str(x) for x in Comb];
##numComb = len(Comb);
##CoexistMat = numpy.zeros((numComb,numComb),numpy.float32);
##connMat = numpy.zeros((numComb,numComb),numpy.float32);
##connDenMat = numpy.zeros((numComb),numpy.float32);
##runs = 80;
##for i in range(runs):
##
##    genes = generateGenes();    
##    sol_genome = decodeGenes(genes);
##
##    print "-weightFns",sol_genome['weightFns'];
##    print "-nodeFns",sol_genome['nodeFns'];
##    print "-weights",sol_genome['connWeights'];
##
##    n = ndm();
##    n.recreateNet(sol_genome);
##
##    
##    CMat = coexistenceLikelihood(n);
##    ConnStrength = coexistenceConnStrength(n);
##    CoexistMat += CMat['inmodel'];
##    connMat += ConnStrength;
##    DenMat = connectionDensity(n);
##    connDenMat += DenMat;
##
##
##
##
##
##visualise_coexist_conn_strength(connMat,CombLabels,show=True);
##visualise_coexistence_likelihood(CoexistMat,CombLabels,show=True);
##visalise_conn_density(connDenMat, CombLabels,show=True);
##hinton_weight(n.connMatrix);
#TEST

#tfcomb,tffreq = getWeightFnUseInfo(n, datasets.DATA1);
#nfcomb,nffreq = getNodeFnUseInfo(n, datasets.DATA1);
#print tfcomb, tffreq;
#print nfcomb, nffreq;
