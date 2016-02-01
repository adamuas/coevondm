from ndm import ndm;
import numpy;
import netParams;
import constants;
import time;
from commons import *;



def randomWeightFn(weightFnSet = None,debug = None):
    """generates a random weight function within the prescribed limits """
    #generate random weight function
    wf = randomWithinParamRange();
    if debug == True:
        if weightFnSet == None:
            wfi = (wf * len(netParams.weightFns));
            print "weightFn:", netParams.weightFns[int(wfi)];
        else:
            wfi = (wf * len(weightFnSet));
            print "weightFn:", weightFnSet[int(wfi)];
    return wf;

def randomNodeFn(nodeFnSet=None,debug = None):
    """generates a ranomd node function within the prescibed limits.
        The function parameters of the node functions are also returned."""
    #generate random node function
    fnParams = [];
    rnd = randomWithinParamRange();
    
    #decode and attach appropriate fn params
    if nodeFnSet == None:
        nfi = (rnd * len(netParams.nodeFns)); #changed
        if nfi > len(netParams.nodeFns):
            nfi -= 1;
        nf = netParams.nodeFns[int(nfi)];
    else:
        #decode from a certain node function set
        nfi = rnd * len(nodeFnSet);
        if nfi > len(nodeFnSet):
            nfi -= 1;
        nf = nodeFnSet[int(nfi)];
        
    #debug
    if debug:
        print "node Fn:", nf;
    
    if(nf == 1): #identity
        #give it a fixed length of the maximum params required
        fnParams = [randomWithinParamRange(),
                    randomWithinParamRange(),
                    randomWithinParamRange()
                    ];
        #debug
        if debug == True:
            print "identity";
            print fnParams;

    if(nf == 2): #sigmoid
        #give it a fixed length of the maximum params required
        fnParams = [randomWithin(netParams.param_limits['sig_minFP'],
                                 netParams.param_limits['sig_maxFP']),
                    randomWithinParamRange(),
                    randomWithinParamRange()
                    ];
        #debug
        if debug == True:
            print "sigmoid";
            print fnParams;

    if(nf == 3): #gaussian
        #give it a fixed length of the maximum params required
        fnParams = [randomWithin(netParams.param_limits['gaus_minFP1'],
                                 netParams.param_limits['gaus_maxFP1']),
                    randomWithin(netParams.param_limits['gaus_minFP2'],
                                 netParams.param_limits['gaus_maxFP2']),
                    randomWithinParamRange()
                    ];
        #debug
        if debug == True:
            print "gauss";
            print fnParams;

    if(nf == 4): #tanh
        #give it a fixed length of the maximum params required
        fnParams = [randomWithin(netParams.param_limits['tanh_minFP'],
                                 netParams.param_limits['tanh_maxFP']),
                    randomWithinParamRange(),
                    randomWithinParamRange()
                    ];
        #debug
        if debug == True:
            print "tanh";
            print fnParams;

    if(nf == 5): #Gaussian II
        fnParams = [randomWithin(netParams.param_limits['gaus_minFP1'],
                                 netParams.param_limits['gaus_maxFP1']),
                    randomWithin(netParams.param_limits['gaus_minFP2'],
                                 netParams.param_limits['gaus_maxFP2']),
                    netParams.param_limits['gaus_thresFP']
                    ];
        #debug
        if debug == True:
            print "gaussII";
            print fnParams;
    
    if(nf == 6): #prob. sigmoid
        #give it a fixed length of the maximum params required
        fnParams = [randomWithin(netParams.param_limits['sig_minFP'],
                                 netParams.param_limits['sig_maxFP']),
                    randomWithinParamRange(),
                    randomWithinParamRange()
                    ];
        #debug
        if debug == True:
            print "prob. sigmoid";
            print fnParams;
        

    return rnd,fnParams;



def randomWithinParamRange(use_NormDistWeigts=False):
    """ return a ranomd number with the """
    
    wDistMean = netParams.wDistMean;
    wDistStd = netParams.wDistStd;
    
    #use weight sampled from a normal distribution
    if use_NormDistWeigts:
        #debug
        if debug:
            print ">>Using Weight Sample from Normal Distribution.";
        
        rnd = numpy.random.normal(wDistMean, wDistStd,1);
    else:
        rnd = numpy.random.rand();

    if(rnd > netParams.param_limits['pMax']):
        rnd = netParams.param_limits['pMax'];
    elif (rnd < netParams.param_limits['pMin']):
        rnd = netParams.param_limits['pMin'];

    return rnd;


def connectFeedforward(num_nodes,node_types, arch_list=None,connActive=None,connWeights= None,use_NormDistWeigts = False):
    """ connects network in feedforward manner """
 
    for i in xrange(num_nodes):
        
        #skip for inactive nodes
        if arch_list[i] != constants.ACTIVE:
            continue;
        
        for j in xrange(num_nodes):
            
                #skip for inactive nodes
                if arch_list[j] != constants.ACTIVE:
                    continue;
                
                
                #only connect to other nodes - not to self
                if(i != j):
                    #input to hidden layer
                    if((node_types[i] == 1 and arch_list[i] == constants.ACTIVE ) and
                       (node_types[j] == 2 and arch_list[j] == constants.ACTIVE)):
                        
                 
                        #make connection active
                        connActive[i][j] = constants.ACTIVE;
                        
                        #generate random weight
                        w = randomWithinParamRange(use_NormDistWeigts);
                        connWeights[i][j] = w;

                        
                    #hidden to output layer
                    if((node_types[i] == 2 and arch_list[i] == constants.ACTIVE ) and
                       (node_types[j] == 3 and arch_list[j] == constants.ACTIVE)):
                        
                        #make connection active
                        connActive[i][j] = constants.ACTIVE;
                        
                        #generate random weight
                        w = randomWithinParamRange(use_NormDistWeigts);
                        connWeights[i][j] = w;
           
       
def decodeTransferFns(genes,num_h_o,start,nodeFnSet=None,weightFnSet=None):
    """decodes the transfer functions of the network """
    
    stop = False;
    point = 0;
    
    #node function, weight function and function params
    nf = 0;
    wf = 0;
    
    #to-return
    wfns = [];
    nfns = [];
    fnParamLst = [];

    if debug == True:
        print "len genes:", len(genes);
        print "start :", start;
        print "num_active hidden:", num_h_o;
        
        
    
    
    for x in xrange(num_h_o):
    
     
        #get weight function
        wfgene = genes[start + point];

        #transFns.append(wfgene);
        if weightFnSet == None:
            wfi = (wfgene * len(netParams.weightFns)); #changed
            if wfi > len(netParams.weightFns):
                wfi = len(netParams.weightFns)-1;
            wf = netParams.weightFns[int(wfi)];
            wfns.append(wf);
        else:
            #use customised set of weight functions
            wfi = (wfgene * len(weightFnSet)); #changed
            if wfi > len(weightFnSet):
                wfi = len(weightFnSet)-1;
            wf = weightFnSet[int(wfi)];
            wfns.append(wf);
            
        #shift pointer
        point += 1;#shift point
        
        #get node function
        nfgene = genes[start + point];
        #transFns.append(nfgene);
        if nodeFnSet == None:
            nfi = (nfgene * len(netParams.nodeFns)); #changed
            if nfi > len(netParams.nodeFns):
                nfi = len(netParams.nodeFns)-1;
            nf = netParams.nodeFns[int(nfi)];
        else:
            nfi = (nfgene * len(nodeFnSet)); #changed
            if nfi > len(nodeFnSet):
                nfi = len(nodeFnSet)-1;
            nf = nodeFnSet[int(nfi)];
        #add transfer function
        nfns.append(nf);

        if(nf == 1): #identity
            point += 4;
            if debug == True:
                print "identity";
            fnParamLst.append([
                                within(genes[start + point -3],
                                netParams.param_limits['pMin'],
                                netParams.param_limits['pMax']),

                                within(genes[start + point -2],
                                netParams.param_limits['pMin'],
                                netParams.param_limits['pMax']),

                                genes[start + point-1]
                            
                            ]);

        if(nf == 2): #sigmoid
            point += 4;
            if debug == True:
                print "sigmoid";
            #only one transfer function
            fnParamLst.append([
                                within(genes[start + point -3],
                                netParams.param_limits['sig_minFP'],
                                netParams.param_limits['sig_maxFP']),

                                within(genes[start + point -2],
                                netParams.param_limits['sig_minFP1'],
                                netParams.param_limits['sig_maxFP1']),

                                genes[start + point-1]

                                ]);

        if(nf ==3): #gaussian
            point += 4;
            if debug == True:
                print "gaussian";
            #only two fn params
            fnParamLst.append([
                                within(genes[start + point -3],
                                netParams.param_limits['gaus_minFP1'],
                                netParams.param_limits['gaus_maxFP1']),

                                within(genes[start + point -2],
                                netParams.param_limits['gaus_minFP2'],
                                netParams.param_limits['gaus_maxFP2']),

                                genes[start + point-1]

                                ]);

        if(nf ==4): #tanh
            point += 4;
            if debug == True:
                print "tanh";
            #only one transfer function
            fnParamLst.append([
                                within(genes[start + point -3],
                                netParams.param_limits['tanh_minFP'],
                                netParams.param_limits['tanh_maxFP']),

                                within(genes[start + point -2],
                                netParams.param_limits['tanh_minFP'],
                                netParams.param_limits['tanh_maxFP']),

                                genes[start + point-1]

            
                                ]);

        if(nf == 5): #gaussin with threshold
            point += 4;
            if debug == True:
                print "gaussian II";
            #only three fn params
            fnParamLst.append([
                                within(genes[start + point -3],
                                netParams.param_limits['gaus_minFP1'],
                                netParams.param_limits['gaus_maxFP1']),

                                within(genes[start + point -2],
                                netParams.param_limits['gaus_minFP2'],
                                netParams.param_limits['gaus_maxFP2']),

                                genes[start + point-1]

                                ]);
        if(nf == 6): #prob. sigmoid
            point += 4;
            if debug == True:
                print "prob. sigmoid";
            #only one transfer function
            fnParamLst.append([
                                within(genes[start + point -3],
                                netParams.param_limits['sig_minFP'],
                                netParams.param_limits['sig_maxFP']),

                                within(genes[start + point -2],
                                netParams.param_limits['sig_minFP1'],
                                netParams.param_limits['sig_maxFP1']),

                                genes[start + point-1]

                                ]);


    return nfns, wfns, fnParamLst;
    