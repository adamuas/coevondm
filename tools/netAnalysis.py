"""
@description: contain functions that analyse a network and return some vital information.
"""

import os, sys, inspect
sys.path.insert(0,"../optim");
sys.path.insert(0,"../");
sys.path.insert(0,"../core");
sys.path.insert(0,"../datasets");
sys.path.insert(0,"../visualisation");


import numpy;

from math import sqrt;
from math import pi;

import ndm;
import datasets;
import commons;
from optimEnDec import *;
import dictWriter;
import csvReader;
import preprocessing;

#import visualiseNet
from visualiseNet import *;
from PyQt4 import QtCore, QtGui

#plot lib
import matplotlib.pyplot as plt;
import matplotlib.animation as anim;
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pylab import *;

import platform;
import netParams;


debug = False;

def tranferFnDist(net = None, net2 = None):
    """ """



def transferFnErrAssoc(net = None, dataset=None):
    """ determines the error contributed by transfer functions of the network """
    
    #weight and node functions
    weightFns = netParams.weightFns;
    nodeFns = netParams.nodeFns;
    
    numI = netParams.nodeConfig['I'];
    numH = netParams.nodeConfig['H'];
    numO = netParams.nodeConfig['O'];
    
    if debug:    
        print "*weightFns Set:",weightFns;
        print "*nodeFns Set:",nodeFns;
    
    #create transfer function combination
    transferFns_err = numpy.zeros((len(weightFns),len(nodeFns)),numpy.float32);
  
        
    #evaluate and get network error
    net_err = max(net.evaluate(dataset));
    
    print "*network error:", net_err;
    
   
   #get output node information 
    out_nodeId = numI + numH ;
    out_node = net.nodes[out_nodeId];
    rel_outId = out_node.map_id;
    if debug:
        print ">>>",out_node.type;
        
    
    
    #determine contribution to error of weight functions
    for node_j in xrange(numH):
        
        #get the hidden node id
        hid_nodeId = node_j + numI;
        node = net.nodes[hid_nodeId];
        
        
        #if the node is an instance of inactive node, skip it.
        if isinstance(node,inactive_node) is True:
            if debug:
                print "inactive node";
            continue;
        
        #get relative node id
        rel_nodeId = node.map_id;
        
        #get weight function
        weightFn = node.weightFn;
        nodeFn = node.nodeFn;
        w_ho = net.connMatrix[rel_nodeId][rel_outId];
        
        #get index of the weight and node fns
        iWeightFn = weightFns.index(weightFn);
        iNodeFn = nodeFns.index(nodeFn);
        
        if debug or False:
            print "-weightFn:",weightFn;
            print "-nodeFn:",nodeFn;
            print "-weight:",w_ho;
          
        #calucate the error associated with the combination
        err_assoc = net_err * w_ho;
        transferFns_err[iWeightFn][iNodeFn] +=  abs(err_assoc);

       
    if debug:
        print "transferFns Error:", transferFns_err;
        
    
    #return the error associated with respective transfer function
    return transferFns_err;
    
    
    
    
    
    
def getTransferFnCombinations(net) :
    """ returns transfer function combinations in matrix form,
    so that they can easily be added to other forms and divided to get a probabilty of
    transfer function selection."""
    
    #weight and node functions
    weightFns = netParams.weightFns;
    nodeFns = netParams.nodeFns;
    
    numI = netParams.nodeConfig['I'];
    numH = netParams.nodeConfig['H'];
    numO = netParams.nodeConfig['O'];
    
    if debug:    
        print "*weightFns Set:",weightFns;
        print "*nodeFns Set:",nodeFns;
    
    #create transfer function combination
    transferFns_comb = numpy.zeros((len(weightFns),len(nodeFns)),numpy.int32);
        
    #determine contribution to error of weight functions
    for node_j in xrange(numH):
        
        #get the hidden node id
        hid_nodeId = node_j + numI;
        node = net.nodes[hid_nodeId];
        
        
        #if the node is an instance of inactive node, skip it.
        if isinstance(node,inactive_node) is True:
            if debug:
                print "inactive node";
            continue;
        
        #get relative node id
        rel_nodeId = node.map_id;
        
        #get weight function
        weightFn = node.weightFn;
        nodeFn = node.nodeFn;
       
        
        #get index of the weight and node fns
        iWeightFn = weightFns.index(weightFn);
        iNodeFn = nodeFns.index(nodeFn);
        
        if debug or False:
            print "-weightFn:",weightFn;
            print "-nodeFn:",nodeFn;
         
        
        #add transfer function combination
        transferFns_comb[iWeightFn][iNodeFn] +=1;
    
    if debug:
        print "transferFns Combination:", transferFns_comb;
        
    
    return transferFns_comb;


def getTransferFnProb(net):
    """ calculates the probability of choosing a transfer function for the given transfer fn combination matrix"""
     #weight and node functions
    numWeightFns = len(netParams.weightFns);
    numNodeFns = len(netParams.nodeFns);
    
    #get transfer function combinations
    transFnComb = getTransferFnCombinations(net);
    
    
    transFnProb = transFnComb * 1/float(numWeightFns*numNodeFns);

    if debug:
        print transFnProb;
    
    return transFnProb;


def getWeightFnUseInfo(net = None, dataset=None):
    """ Returns the associated error and frequency of use for all given weight function """
    
    #weight and node functions
    weightFns = netParams.weightFns;
    nodeFns = netParams.nodeFns;
    
    numI = netParams.nodeConfig['I'];
    numH = netParams.nodeConfig['H'];
    numO = netParams.nodeConfig['O'];
    
    if debug:    
        print "*weightFns Set:",weightFns;
        print "*nodeFns Set:",nodeFns;
    
    #create transfer function combination
    weightFns_err = numpy.zeros((len(weightFns)),numpy.float32);
    weightFns_freq = numpy.zeros((len(weightFns)),numpy.int32);
        
    #evaluate and get network error
    net_err = max(net.evaluate(dataset));
    
    print "*network error:", net_err;
    
   
   #get output node information 
    out_nodeId = numI + numH ;
    out_node = net.nodes[out_nodeId];
    rel_outId = out_node.map_id;
    if debug:
        print ">>>",out_node.type;
        
    
    
    #determine contribution to error of weight functions
    for node_j in xrange(numH):
        
        #get the hidden node id
        hid_nodeId = node_j + numI;
        node = net.nodes[hid_nodeId];
        
        
        #if the node is an instance of inactive node, skip it.
        if isinstance(node,inactive_node) is True:
            if debug:
                print "inactive node";
            continue;
        
        #get relative node id
        rel_nodeId = node.map_id;
        
        #get weight function
        weightFn = node.weightFn;
        
        w_ho = net.connMatrix[rel_nodeId][rel_outId];
        
        #get index of the weight
        iWeightFn = weightFns.index(weightFn);
        
        
        if debug or False:
            print "-weightFn:",weightFn;
            print "-weight:",w_ho;
        
        #add transfer function combination
        if weightFns_freq[iWeightFn] < 1:
            weightFns_freq[iWeightFn] +=1;
        
        #calucate the error associated with the combination
        err_assoc = net_err * w_ho;
        weightFns_err[iWeightFn] +=  err_assoc;

    if debug:
        print "weightFns Freq:", weightFns_freq;
        print "weightFns Error:", weightFns_err;
        
    
    #return the error associated with respective transfer function
    return weightFns_err,weightFns_freq;
    
    
    
def getWeightFnProb():
    """ Returns the probability of using the given weight function """
    
     #weight and node functions
    numWeightFns = len(netParams.weightFns);
    numNodeFns = len(netParams.nodeFns);
    
    #get transfer function combinations
    weightFnErrAssoc, weightFnFreq = getWeightFnUseInfo(net);
    
    
    weightFnProb = weightFnFreq * 1/float(numWeightFns);

    if debug:
        print transFnProb;
    
    return transFnProb;
    
    

def getNodeFnUseInfo(net = None, dataset=None):
    """ Returns the associated error and frequency of use for all given weight function """
    
    #weight and node functions
    weightFns = netParams.weightFns;
    nodeFns = netParams.nodeFns;
    
    numI = netParams.nodeConfig['I'];
    numH = netParams.nodeConfig['H'];
    numO = netParams.nodeConfig['O'];
    
    if debug:    
        print "*weightFns Set:",weightFns;
        print "*nodeFns Set:",nodeFns;
    
    #create transfer function combination
    nodeFns_err = numpy.zeros((len(nodeFns)),numpy.float32);
    nodeFns_freq = numpy.zeros((len(nodeFns)),numpy.int32);
        
    #evaluate and get network error
    net_err = max(net.evaluate(dataset));
    
    print "*network error:", net_err;
    
   
   #get output node information 
    out_nodeId = numI + numH ;
    out_node = net.nodes[out_nodeId];
    rel_outId = out_node.map_id;
    if debug:
        print ">>>",out_node.type;
        
    
    
    #determine contribution to error of weight functions
    for node_j in xrange(numH):
        
        #get the hidden node id
        hid_nodeId = node_j + numI;
        node = net.nodes[hid_nodeId];
        
        
        #if the node is an instance of inactive node, skip it.
        if isinstance(node,inactive_node) is True:
            if debug:
                print "inactive node";
            continue;
        
        #get relative node id
        rel_nodeId = node.map_id;
        
        #get weight function
        weightFn = node.weightFn;
        nodeFn = node.nodeFn;
        
        w_ho = net.connMatrix[rel_nodeId][rel_outId];
        
        #get index of the node fns
        iNodeFn = nodeFns.index(nodeFn);
        
        if debug or False:
            print "-NodeFn:",nodeFn;
            print "-weight:",w_ho;
        
        #add transfer function combination
        if nodeFns_freq[iNodeFn] < 1:
            nodeFns_freq[iNodeFn] +=1;
        
        #calucate the error associated with the combination
        err_assoc = net_err * w_ho;
        nodeFns_err[iNodeFn] +=  err_assoc;

    if debug:
        print "NodeFns Freq:", nodeFns_freq;
        print "NodeFns Error:", nodeFns_err;
        
    
    #return the error associated with respective transfer function
    return nodeFns_err,nodeFns_freq;
    
    
    
def getNodeFnProb(net = None, dataset=None):
    """ Returns the probability of using the given weight function """
    
     #weight and node functions
    numWeightFns = len(netParams.weightFns);
    numNodeFns = len(netParams.nodeFns);
    
    #get transfer function combinations
    nodeFnErrAssoc, nodeFnFreq = getNodeFnUseInfo(net,dataset);
    
    
    nodeFnProb = nodeFnFreq * 1/float(numNodeFns);

    if debug:
        print "probability of node function: ", nodeFnProb;
    
    return nodeFnProb;
    
    
    
##TEST
#genes = generateGenes();    
#sol_genome = decodeGenes(genes);
##
##print "-weightFns",sol_genome['weightFns'];
##print "-nodeFns",sol_genome['nodeFns'];
##print "-weights",sol_genome['connWeights'];
#
#n = ndm();
#n.recreateNet(sol_genome);
#
#tfcomb,tffreq = getWeightFnUseInfo(n, datasets.DATA1);
#nfcomb,nffreq = getNodeFnUseInfo(n, datasets.DATA1);
#print tfcomb, tffreq;
#print nfcomb, nffreq;

    
