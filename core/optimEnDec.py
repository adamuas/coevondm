#imports
from ndm import ndm;
import numpy;
import netParams;
from commons import *;
import constants;
import math;
import logging;


"""
@description: The encoder uses direct encoding to encode all information of the network
WHO TO CONTACT(In case of codes that are not very clear or a bug):
@author: Abdullahi
@author email : abdullah.adam89@hotmail.com
* Add new node function : You will have to also modify this file, particularly the generateGenes and the decodeTransferFns().
"""
#Concept:
#-encoding by node, each nodes information is encoded onto the genetic string.
#-similar parameters are kept close to each other on the genestring
#-different paramters are encoded with some distance

#Switch the Debug Attribute to True to see what happends as this module works.
#It's also useful for debugging.
debug = False;

def generateGenes(nodeFns = None, weightFns = None, use_NormDistWeigts = False, float32 = True):
    """ Generate optimised genes"""
    
    #debug
    if debug:
        print "#Generating Genes";
    
    #SETTINGS
    #probability of having an active hidden node
    prbActiveHidNode = netParams.probHidNodeCreation;
    
    #genetic string
    #include cost0(train), cost1(test), misc param, and age
    genetic_string  = [-1, -1, 0.0, 0];

    #get total nodes - required, maximum
    num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
    
    """(1) Architecture """
    arch_list = [];
    node_types = [];
    #encod input nodes statust
    for x_i in xrange(netParams.nodeConfig['I']):
        node_types.append(1); #input node
        arch_list.append(1.0);
        
    #hidden nodes
    #add atleast one node then every other node on probablistic basis
    hid_nodes = [];
    for x in range(netParams.nodeConfig['H']):
        node_types.append(2); #hidden node
        if(len(hid_nodes) == 0):
            hid_nodes.append(1.0);
        else:
            if(numpy.random.rand() <= prbActiveHidNode):
                hid_nodes.append(1.0);
            else:
                hid_nodes.append(0.0);
    #append to arch list
    arch_list.extend(hid_nodes);
    #output nodes
    for x_i in xrange(netParams.nodeConfig['O']):
        node_types.append(3); #hidden node
        arch_list.append(1.0);

    #append to genes
    genetic_string.extend(arch_list);

    #debug
    if debug == True:
        print "-architecture"
        print arch_list;
    
    """ ACTIVE NODE COUNT """
    num_active_nodes = arch_list.count(constants.ACTIVE);
    
    
    """(2) Transfer functions and their parameters """
    
    #create transfer function and params based on activness of the node
    trans_fns = [];

    nf = 1;  #node fn - Identity
    wf = -1; #weight fn - none
    fnParams = [];
    
# Do not encode for input nodes , Reason: Their values are going to be same always anyway.
    for h_o in xrange(netParams.nodeConfig['H'] + netParams.nodeConfig['O']):
        
        #get relative node index
        node_idx = h_o + netParams.nodeConfig['I'];
        
        #only create node information if its active
        if arch_list[node_idx] == constants.ACTIVE:
            #get random node function
            wf = randomWeightFn(weightFns);
            #encode weight function
            trans_fns.append(wf);
    
            #get random node function
            nf,fnParams = randomNodeFn(nodeFns);
            #encode node function
            trans_fns.append(nf);
            #encode transfer functions
            trans_fns.extend(fnParams);


    #transfer functions
    #debug
    if debug == True:
        print "-transFns"
        print trans_fns;

    #encode
    genetic_string.extend(trans_fns);
    

    """(3)Connectivity and weight matrix - feedforward"""
    
    if float32 == True:
        connActive = numpy.zeros((num_nodes,num_nodes),numpy.float32);
        connWeights = numpy.zeros((num_nodes,num_nodes),numpy.float32);
    else:
        connActive = numpy.zeros((num_nodes,num_nodes),numpy.float16);
        connWeights = numpy.zeros((num_nodes,num_nodes),numpy.float16);

    #connect in feedforward manner
    connectFeedforward(num_nodes,node_types,arch_list,connActive,connWeights);
    
    #optimise connection matrix and connection weights
    #-only connections for active nodes would be left
    connActive = getActiveConnMat(num_active_nodes,arch_list,connActive) ;
    connWeights = getActiveWeightMat(num_active_nodes,arch_list,connWeights);
    
    #debug
    if debug == True:
        print "-connections active";
        print connActive;
        print "-connWeights active";
        print connWeights;
        
    #encode
    genetic_string.extend(connActive.flatten().tolist());
    genetic_string.extend(connWeights.flatten().tolist());

    #debug
    if debug == True:
        print "-genetic_string";
        print genetic_string;

    

    """(4) Weight bias and autoweight """
    bias =[];
    biasWeights = [];

    for i in xrange(num_nodes):
        if node_types[i] == 1 : #input node
            bias.append(0.0);
            biasWeights.append(1.0);
        else:
            if(arch_list[i] == constants.ACTIVE):
                bias.append(randomWithinParamRange());
                biasWeights.append(randomWithinParamRange());

    #debug
    if debug == True:
        print "weight bias";
        print bias;
        print "autoweights";
        print biasWeights;

    #enode
    genetic_string.extend(bias);
    genetic_string.extend(biasWeights);

    #debug
    if debug == True:
        print genetic_string;
        print "-complete!!!";

    return genetic_string;


def decodeGenes(genes,nodeFns=None,weightFns=None):
    """ decodes the genes encoded """
    
    #debug
    if debug:
        print "#Choromosome information"
        print "-length:", len(genes);
        print "genes to decode", genes;
        
    #precaution
    if len(genes) == 0:
        print "-No choromosome to decode";
        return None;

    #dictionary of genomes to be returned
    genome = dict();
    
    #decode the cost and age
    genome['cost'] = genes[constants.COST_GENE];
    genome['cost2'] = genes[constants.COST2_GENE];
    genome['misc'] = genes[constants.MISC_GENE];
    genome['age'] = genes[constants.AGE_GENE];
    gene_start = constants.META_INFO_COUNT;
    
    #debug
    if debug == True:
        print "cost",genome['cost'];
        print "cost2",genome['cost2'];
        print "misc",genome['misc'];
        print "age",genome['age'];
        
    #get total nodes - required, maximum
    num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];

    """(1) decode architecture """
    genome['architecture'] = genes[gene_start:num_nodes+gene_start];
    #do some gene correction to make sure they are valid genes -
    #This is necessary because of the mutations and other genetic operators might cause abnormalities
    realToBinary(genome['architecture']);
    
    #debug
    if debug == True:
        print "Architecture";
        print genome['architecture'];
    
     #count number of active genes
    num_active_nodes = genome['architecture'].count(constants.ACTIVE);
    
    """(2) Decode transfer functions and their parameters """
    num_active_h_o = num_active_nodes - netParams.nodeConfig['I'];
    
    gene_start += num_nodes;#shift start point
    
    #get the transfer and weight functions
    nfns,wfns,fnParams = decodeTransferFns(genes,num_active_h_o, gene_start,nodeFns,weightFns);
    
    #maintain original lengths for decoding next genes
    len_nfns = len(nfns);
    len_wfns = len(wfns);
    len_fnParams = lenInnerList(fnParams);
    genome['nodeFns']  = nfns;
    genome['weightFns']  = wfns;
    genome['fnParams']  = fnParams;
    

    #debug
    if debug == True:
        print "transfer fn.";
        print "nodeFns",genome['nodeFns'];
        print "weightFns",genome['weightFns'];
        print "fnParams",genome['fnParams'];
    
    
    """(3) decode connectivity and weight matrix """
    #get the genes for connection status
    gene_start  += (len_nfns+len_wfns+len_fnParams);
    tot_conn = num_active_nodes * num_active_nodes;
    connectivity = genes[gene_start:tot_conn+gene_start];
    
    #correct any gene abnormalities
    connectivity = realToBinary(connectivity);
    
    #make matrix form for use in network
    genome['connectivity'] = matrixForm(connectivity, num_active_nodes, num_active_nodes);
    
    #debug
    if debug == True:
        print "connectivity";
        print genome['connectivity'];
    
    #get weights
    gene_start += tot_conn; #shift start point
    connWeights = genes[gene_start:tot_conn+gene_start];
    genome['connWeights'] = matrixForm(connWeights, num_active_nodes, num_active_nodes);
    #debug
    if debug == True:
        print "connWeights";
        print genome['connWeights'];
        
        
    """(4)bias and its weights """
    #get bias
    gene_start += tot_conn; #shift start point
    genome['bias']  = genes[gene_start:num_active_nodes+gene_start];

    #gets its weights
    gene_start += num_active_nodes; #shift start point
    genome['biasWeights']  = genes[gene_start:];


    #debug
    if debug == True:
        print "-bias and its weights"
        print genome['bias'];
        print genome['biasWeights'];

    #return the genome
    return genome;
    

def genomeToGenes(genome,nodeFns=None,weightFns=None):
    """ encodes the genome to genes """
    
    #debug
    if debug:
        print "#Genome information"
        print "-length:", len(genome);
        print "genome ", genome;
        
    #precaution
    if genome == None:
        print "-No genome to encode";
        return None;
    
    #genes
    genes = [];
    
    #decode the cost and age
    genes.append(genome['cost']);
    genes.append(genome['cost2']);
    genes.append(genome['misc']);
    genes.append(genome['age']);
    
    gene_start = constants.META_INFO_COUNT;

    """(1) encode architecture """
    genes.extend(genome['architecture']);
    
    #debug
    if debug == True:
        print "Architecture";
        print genome['architecture'];
        
    """(2)  encode transfer functions and their parameters  """
    nodeFns = genome['nodeFns'];
    weightFns = genome['weightFns'];
    fnParams = genome['fnParams'];
    
    
    #debug
    if debug == True:
        print "transfer fn.";
        print "nodeFns",genome['nodeFns'];
        print "weightFns",genome['weightFns'];
        print "fnParams",genome['fnParams'];
    
    #encode transfer function genes
    tfgenes = encodeTransferFns(weightFns,nodeFns,fnParams);
    
    #append genes
    genes.extend(tfgenes);
    
   
   
    """(3) decode connectivity and weight matrix """
    connectivity = matrixToList(genome['connectivity'].tolist());
    genes.extend(connectivity);
    
    #debug
    if debug == True:
        print "connectivity";
        print genome['connectivity'];
    
    connWeights = matrixToList(genome['connWeights'].tolist());
    genes.extend(connWeights);
    
    #debug
    if debug == True:
        print "connWeights";
        print genome['connWeights'];
        
    """(4)bias and its weights """
    bias = genome['bias'];
    biasWeights = genome['biasWeights'];
    
    #add to genes
    genes.extend(bias);
    genes.extend(biasWeights);
    
    return genes;
    
        
def decodeWeights(genes):
    """ decodes the weights of a genetic string """
    #dictionary of genomes to be returned
    
    gene_start = constants.META_INFO_COUNT;
    gene_end = 0;
        
    #get total nodes - required, maximum
    num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];

    """(1) decode architecture """
    arch = genes[gene_start:num_nodes+gene_start];
    #do some gene correction to make sure they are valid genes -
    #This is necessary because of the mutations and other genetic operators might cause abnormalities
    realToBinary(arch);
    gene_start += num_nodes;#shift start point
    
    #debug
    if debug == True:
        print "Architecture";
        print arch;
    
     #count number of active genes
    num_active_nodes = arch.count(constants.ACTIVE);
    num_active_h_o = num_active_nodes - netParams.nodeConfig['I'];
    tot_conn = num_active_nodes * num_active_nodes;
    #calculate number of transfer function genes
    
    gene_start += (num_active_h_o * constants.TRANS_INFO_COUNT) ;
    #connectivity
    connectivity = realToBinary(genes[gene_start: tot_conn+gene_start]);
    
    #shift
    gene_start = gene_start + tot_conn;
    gene_end = gene_start + tot_conn;
    
    #weights
    weights = genes[gene_start:gene_end];
    
    #only active connection weights
    weights = numpy.array(weights) * numpy.array(connectivity);
    
    return weights;
    

    
def encodeTransferFns(wfns,nfns,fnparams):
    """encodes the transfer functions of a network  """
    
    genes = [];
    
    if debug :
        print "wfns",wfns;
        print "nfns",nfns;
        print "fparams",fnparams;
    
    for wfi,wf in enumerate(wfns):
        
        #node fn
        nf = encodeNodeFn(nfns[wfi]);
        wf = encodeWeightFn(wf);
        fnp = fnparams[wfi];
        
        if debug :
            print "nf",nf;
            print "wf",wf;
            print "fnp",fnp;
        
        genes.append(wf);
        genes.append(nf);
        genes.extend(fnp);
        
    
    
    return genes;
    
    

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
        
        
        #logging.info('number_active_hidden %d', num_h_o);
    
    
    for x in xrange(num_h_o):
    
     
        #get weight function
        wfgene = genes[start + point];

        #transFns.append(wfgene);
        if weightFnSet == None:
            wfi = (wfgene * len(netParams.weightFns)); #changed
            if wfi > len(netParams.weightFns):
                wfi = len(netParams.weightFns)-1;
            wf = netParams.weightFns[int(wfi)];
            
        else:
            #use customised set of weight functions
            wfi = (wfgene * len(weightFnSet)); #changed
            if wfi > len(weightFnSet):
                wfi = len(weightFnSet)-1;
            wf = weightFnSet[int(wfi)];
            
        #add weight function    
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
    

def connectFeedforward(num_nodes,node_types, arch_list,connActive,connWeights,use_NormDistWeigts = False):
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
           
       

    
def getActiveConnMat(num_active_nodes,arch_list,connMat):
        """gets the connections possible from the given nodes that are active """

   
        #get connectivity possible
        connPossible = numpy.zeros((num_active_nodes,num_active_nodes),numpy.float32);
        #init pointers
        row_idx = 0;
        col_idx = 0;
        connMat_idx = 0;
        for row in connMat:
            row_list = [];
            col_idx = 0; #re-init
            for col in row:
                #copy only connections of active nodes
                if(arch_list[row_idx] != 0):
                    if(arch_list[col_idx] != 0):
                        #cut out only possible connections
                        row_list.append(col);
                #icreament col
                col_idx += 1;

            #copy only if its active
            if(arch_list[row_idx] != 0):
                connPossible[connMat_idx] = row_list;
                connMat_idx += 1;
                    
            #increament
            row_idx += 1;

        #debug
        if debug == True:
            print "optim conn status mat:"
            print connPossible; 

        return connPossible;

def getActiveWeightMat(num_active_nodes,arch_list,connWeights):
        """ returns the feasible weight matrix """


        #get connectivity possible
        weightsPossible = numpy.zeros((num_active_nodes,num_active_nodes),numpy.float32);
        #init pointers
        row_idx = 0;
        col_idx = 0;
        connMat_idx = 0;
        for row in connWeights:
            row_list = [];
            col_idx = 0; #re-init
            for col in row:
                #copy only connections of active nodes
                if(arch_list[row_idx] != 0):
                    if(arch_list[col_idx] != 0):
                        #cut out only possible connections
                        row_list.append(col);
                #icreament col
                col_idx += 1;

            #copy only if its active
            if(arch_list[row_idx] != 0):
                weightsPossible[connMat_idx] = row_list;
                connMat_idx += 1;
                    
            #increament
            row_idx += 1;

        #debug
        if debug == True:
            print "optim conn weight mat:"
            print weightsPossible;

        return weightsPossible;

def encodeToGene(val, lenght):
    """encodes to genetic string - for weight and node function encoding """
    val = (float(val) /lenght-1);
    if debug == True:
        print "gene:", val;
    return val;

def encodeWeightFn(wf,weightFnSet = None):
    """encodes the weight function """
    
    if weightFnSet == None:
        wf = (wf / float(len(netParams.weightFns)));
    else:
        wf = (wf / float(len(weightFnSet)));

    return wf;
        
    

def randomWeightFn(weightFnSet = None):
    """generates a random weight function within the prescribed limits """
    #generate random weight function
    wf = randomWithinParamRange();
    
    if weightFnSet == None:
        wfi = (wf * len(netParams.weightFns));
        
        #print "weightFn:", netParams.weightFns[int(wfi)];
    else:
        wfi = (wf * len(weightFnSet));
        #print "weightFn:", weightFnSet[int(wfi)];
        
    return wf;

def encodeNodeFn(nf,nodeFnSet=None):
    """ returns the encoded value"""
    
    #decode and attach appropriate fn params
    if nodeFnSet == None:
        nf = (nf / float(len(netParams.nodeFns))); #changed 
       
    else:
        #decode from a certain node function set
        nf = nf / float(len(nodeFnSet));
        
    return nf;
        
    
def randomNodeFn(nodeFnSet=None):
    """generates a ranomd node function within the prescibed limits.
        The function parameters of the node functions are also returned."""
    #generate random node function
    fnParams = [];
    rnd = randomWithinParamRange();
    
    #decode and attach appropriate fn params
    if nodeFnSet == None:
        nfi = (rnd * len(netParams.nodeFns)); #changed 
        nf = netParams.nodeFns[int(nfi)];
    else:
        #decode from a certain node function set
        nfi = rnd * len(nodeFnSet);
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