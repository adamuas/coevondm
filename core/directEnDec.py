#imports
from ndm import ndm;
import numpy;
import netParams;
from commonEnDec import *;
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


#Switch the Debug Attribute to True to see what happends as this module works.
#It's also useful for debugging.
debug = True;


def generateGenes(nodeFns = None, weightFns = None, use_NormDistWeigts = False, float32 = True):
    """ generates the genes using direct encoding """
    
    #debug
    if debug:
        print "#Generating Genes";
    
    #SETTINGS
    #probability of having an active hidden node
    prbActiveHidNode = netParams.probHidNodeCreation;
    
    #genetic string
    #include cost(train),cost(test),contribution potential and age
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
    for x in xrange(netParams.nodeConfig['H']):
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
    
    #debug
    if debug == True:
        print "-connections active"
        print connActive;
        print "-connWeights active"
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
    if debug  or False:
        print genetic_string;
        print "-complete!!!";

    return genetic_string;


def decodeGenes(genes,nodeFns=None,weightFns=None):
    """ decodes the genes encoded - only from direct encoded genes """
    
    #debug
    if debug:
        print "#Choromosome information"
        print "\n-length:", len(genes);
        print "\ngenes to decode", genes;
        
    #precaution
    if len(genes) == 0:
        print "-No choromosome to decode";
        return None;

    #dictionary of genomes to be returned
    genome = dict();
    
    #decode the cost and age
    genome['cost'] = genes[constants.COST_GENE];  #train error
    genome['cost2'] = genes[constants.COST2_GENE]; #test error
    genome['contrb'] = genes[constants.CONTRB_GENE]; #contribution to complimentarity
    genome['age'] = genes[constants.AGE_GENE]; #age of the gene
    gene_start = constants.META_INFO_COUNT;
    
    #debug
    if debug == True:
        print "#META INFORMATION";
        print "cost(train)", genome['cost'];
        print "cost (test)",genome['cost2'];
        print "contrib potential",genome['contrb'];
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
    #num_active_nodes = genome['architecture'].count(constants.ACTIVE);
    
    
    """(2) Decode transfer functions and their parameters """
    num_h_o =  netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
    
    gene_start += num_nodes;#shift start point
    
    #get the transfer and weight functions
    nfns,wfns,fnParams = decodeTransferFns(genes,num_h_o, gene_start,nodeFns,weightFns);
    
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
    tot_conn = num_nodes * num_nodes;
    connectivity = genes[gene_start:tot_conn+gene_start];
    
    #correct any gene abnormalities
    connectivity = realToBinary(connectivity);
    
    #make matrix form for use in network
    genome['connectivity'] = matrixForm(connectivity, num_nodes, num_nodes);
    
    #debug
    if debug == True:
        print "connectivity";
        print genome['connectivity'];
    
    #get weights
    gene_start += tot_conn; #shift start point
    connWeights = genes[gene_start:tot_conn+gene_start];
    genome['connWeights'] = matrixForm(connWeights, num_nodes, num_nodes);
    #debug
    if debug == True:
        print "connWeights";
        print genome['connWeights'];
        
        
    """(4)bias and its weights """
    #get bias
    gene_start += tot_conn; #shift start point
    genome['bias']  = genes[gene_start:num_nodes+gene_start];

    #gets its weights
    gene_start += num_nodes; #shift start point
    genome['biasWeights']  = genes[gene_start:];


    #debug
    if debug == True:
        print "-bias and its weights"
        print genome['bias'];
        print genome['biasWeights'];

    #return the genome
    return genome;
