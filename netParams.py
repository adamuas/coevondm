# Network Parameters
""" Problem type"""
#1- REGRESSION
#2- CLASSIFICATION
problem_type = 2;
""" Status Messages """
verbose = False;
debug = False;
""" Network Specificaions """
iterations = 5;

""" Net param limits """
param_limits = {
    'minWB': 0.0,
    'maxWB': 1.0,
    'sig_minFP': 0.0,
    'sig_maxFP': 1.0,
    'sig_minFP1': 0.0,
    'sig_maxFP1': 1.0,
    'tanh_minFP':0.0,
    'tanh_maxFP':1.0,
    'gaus_minFP1':0.0,
    'gaus_maxFP1':1.0,
    'gaus_minFP2':0.0,
    'gaus_maxFP2':1.0,
    'gaus_thresFP':0.3,
    'pMin':0.0,
    'pMax':1.0
    };
"""connection limits"""
conn_limits  = {
    'inToIn':1,
    'inToMid':1,
    'inToOut':1,
    'midToMid':1,
    'midToIn':1,
    'midToOut':1,
    'outToOut':1,
    'outToMid':1,
    'outToIn': 1
    };

""" Weight and node functions"""
weightFns = [1,2,3,4,5,6,7];
nodeFns = [1,2,3,4,5];

#transfer function complexification
use_initTransFns = False;
initWeightFns = [2];
initNodeFns = [1,2];

#complexification criteria
FIXED_INTERVAL = 1;
STAGNATION = 2;
complexifyCriteria = 1; #complexification criteria.
complexifyInterval = 20;
stagnationLength = 10;
complexifyMethod = 1;

#Transfer function control
use_cardinal = False; #use only the cardinal transfer functions
nodeFnsToUse = 1; #defines the cardinality of the node functions being used
weightFnsToUse = 1; #defines the cardinality of the weight functions being used
#use nomally distributed weights
use_NormDistWeigts = False;
wDistMean = 0.0;
wDistStd = 0.1;


nodeFnParamCounts = {'1':0, 
                     '2':1,
                     '3':2,
                     '4':1,
                     '5':3
                     }; #node function params required

"""Connections"""
maxHidLayers = 3; #layers 
nodeConfig = { 'I': 4,
               'H': 4,
               'O': 1}; #configuration of nodes

""" Input distortions/noise """
use_gauss_noise = True ;#use gaussian noise on the inputs
use_nonlinear_trans = False; #use non linear transformation on inputs.
use_random_sample_wreplacement = False; #use bagging method to train the neural networks from random samples with replacement.

"""Node relationship"""
#functions
#-1. Standard dev.
#-2. Summ of Euclid.
use_nodeRel = False;
relational_fns = [1, 2];

"""Context layer - elman (for recurrent connectivity)"""
use_context = True;

""" network development """
probHidNodeCreation = 0.35;
probInToHidConn = 0.9;
probHidToOutConn = 0.9;
