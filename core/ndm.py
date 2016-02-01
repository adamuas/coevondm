"""
@description: Class describing neural diversity machines.
@author : Abdullahi S. Adamu
@email : abdullah.adam89@hotmail.com
"""
import numpy;


"""imports"""
#import pycuda.gpuarray as gpuarray;
#import pycuda.driver as cuda;
#import pycuda.autoinit;
#import numpy;
#from pycuda.compiler import SourceModule;

#import the network parameters
import netParams;

#debug tool
import logging;
import warnings;
logging.basicConfig(filename='logger.log',level=logging.DEBUG)


#import constants
import constants;

#import node class
from node import *;


#math class
import math;
import random;

#copy
import copy;



"""
NOTE:
 - to adjust the parameters of the node
 -
"""

"""
UPDATES:
- added new transfer functions which need to be integrated and tested 

"""

##weight functions##
# 1- INNER_PROD
# 2- EUCLID_DIST 
# 3- HIGHER_ORDER_PROD
# 4- HIGHER_ORDER_SUB 
# 5- STD_DEV
# 6- MIN 
# 7- MAX 


##node functions##
# 1 - Identity
# 2 - Sigmoid
# 3 - Gaussian
# 4 - Tanh (hyperbolic tangent)
# 5 - Gaussian II (thresholded)
# 6 - Probablistic Sigmoid

#Relational Node Functions
# 1 - standard deviation
# 2 - Sum of euclidean distances


#ISSUE(s)
# SQUASH: for output nodes with inner product and identity node functions(SOLVED)
# MISJUDGED OUTPUTS:  multiple outputs while maximum is one (SOLVED)



# NOTES
# * A new node function - the hyperbolic tangent has been implemented.
# * Standard Deviation - was implemented differently, it performs an inner-product then checks standard deviation of the input potentials
# * Relational Nodes - have been implemented partially, the only part that is left is to implement a function to find relationship and inject into hidden nodes as inputs.
#       There would be one connection to each hidden node, and the weight would be set to one.
#       Optimisation should be able optimise this by pruning connections and tuning weights.
# * float16 - the precision for real values is "Half precision float: sign bit, 5 bits exponent, 10 bits mantissa".
#       This is to reduce the dimension of the search space.




class ndm:


    def __init__(self):
        """Meta-info """
        self.fitness = 1.0;
        """ Problem type"""
        self.problem_type = netParams.problem_type;
        """ Status"""
        self.ready = False;
        self.verbose = netParams.verbose;
        self.debug = netParams.debug;
        """ Network Specificaions """
        self.iterations = 1;
        """ Net param limits """
        self.param_limits = {
            'minWB': netParams.param_limits['minWB'],
            'maxWB': netParams.param_limits['maxWB'],
            'sig_minFP': netParams.param_limits['sig_minFP'],
            'sig_maxFP': netParams.param_limits['sig_maxFP'],
            'tanh_minFP':netParams.param_limits['tanh_minFP'],
            'tanh_maxFP':netParams.param_limits['tanh_maxFP'],
            'gaus_minFP1': netParams.param_limits['gaus_minFP1'],
            'gaus_maxFP1': netParams.param_limits['gaus_maxFP1'],
            'gaus_minFP2': netParams.param_limits['gaus_minFP2'],
            'gaus_maxFP2': netParams.param_limits['gaus_maxFP2'],
            'gaus_thresFP': netParams.param_limits['gaus_thresFP'],
            'pMin': netParams.param_limits['pMin'],
            'pMax': netParams.param_limits['pMax']
            };
        
        """connection limits""" # Connection LIMITS Have not been implemented (Free connectivity desired)
        self.conn_limits  = {
            #'inToIn': netParams.conn_limits['inToIn'],
            'inToMid':netParams.conn_limits['inToMid'],
            'inToOut':netParams.conn_limits['inToOut'],
            'midToMid':netParams.conn_limits['midToMid'],
            #'midToIn':netParams.conn_limits['midToIn'],
            'midToOut':netParams.conn_limits['midToOut'],
            #'outToOut':netParams.conn_limits['outToOut'],
            #'outToMid':netParams.conn_limits['outToMid'],
            #'outToIn': netParams.conn_limits['outToIn']
            };
        """Connections - others"""
        self.maxHidLayers = netParams.maxHidLayers; #layers 
        self.connMatrix = []; #connection matrix
        self.conn_active = []; #active matrix
        
        """ input manipulation """
        self.use_gauss_noise = netParams.use_gauss_noise; #inject gaussian noise into the dataset
        self.use_nonlinear_trans = netParams.use_nonlinear_trans; #transform input by a random function on inputs.
        self.use_random_sample_wreplacement = netParams.use_random_sample_wreplacement; #Bagging
        
        """ Nodes"""
        self.nodes = []; #nodes-stores node information
        self.nodes_active = []; #nodes status : on or off
        self.outputMatrix = []; #matrix of node outputs
        self.biasMatrix = []; #node bias values
        self.autoWeightMatrix = []; #auto weights values for nodes
        self.nodeConfig = netParams.nodeConfig;
        self.weightFns = netParams.weightFns;
        self.nodeFns = netParams.nodeFns;
        self.outputHistory = [];

        
        """ NETWORK GENOME -
        It encodes information of the network, and is able to generate the genes of the network.
        In addition, it can also changes made to the genes made during optimisation.
        """
        self.genome = dict();
        

        #higher order product
        self.prodConstant = 3;
        
        """ Context Layer -
        The context layer helps to store the state of the hidden units for use in the next time frame,
        this improves the neural networks ability to predict - The Elman Architecture is used.
        """
        self.use_context = netParams.use_context;
        self.contextMatrix = []; #saves the values of the hidden nodes at time intervals
        self.contextWeights = []; #weights between the context layer and the

        """ Relational Nodes - 
        The role of these nodes it to establish relationship between the inputs being fed to the network, in hopes of improvement in
        performance.
        """
        self.use_nodeRel = netParams.use_nodeRel;
        self.num_rel_nodes = 1; #number of relational nodes
        self.relaltional_fns = netParams.relational_fns;
        self.nodeRelMat = []; #output of the relational nodes 
        self.nodeRelWeights = [];#weights of relational nodes to hidden layer
        

    def setIteration(self,i):
        """sets the number of iterations"""
        self.iteration = i;

    def getIteration(self):
        """returns the number of iterations"""
        return self.iteration;

    def setNodeConfig(self, nodeConfig):
        """sets the node configuration"""
        self.nodeConfig = nodeConfig;
        
    def getNodeConfig(self):
        """sets the node configuration"""
        return self.nodeConfig;

    def setMaxLayers(self,n):
        """ sets the number of layers"""
        self.maxHidLayers = n;
        
    def getMaxLayers(self):
        """ sets the number of layers"""
        return self.maxHidLayers;

    
    def recreateNet(self, genome = None):
        """ recreates network from genome"""
        
        #log
        #logging.info('*Recreate Net');
        
        #check if there is a given genome        
        if genome != None:
            self.genome = copy.deepcopy(genome);
           

        #clear any old nodes -if any
        self.nodes = [];


        #get number of active nodes
        activNodeCount  = self.getCountActivNodes(genome);
        
        #count nodes by configuration
        nodeCount = self.nodeConfig['I'] + self.nodeConfig['H'] + self.nodeConfig['O'];
        
        

        # RECREATE NETWORK FROM GENOME
        """ META INFO"""
        self.fitness = self.genome['cost'];

        """(1) Architecture """
        """ Node status Matrix """
        #recreate node status matrix
        self.nodes_active = self.genome['architecture'];
        self.addedNodeActivity = numpy.ones((genome['architecture'].count(constants.ACTIVE)));
        if self.verbose == True:
            print "-architecture matrix created.";
            print self.nodes_active;
            


        """(2) connectivity and weight matrix """   
        """ Connection  Weights Matrix """
        #recreate connection matrix
        self.connMatrix = self.genome['connWeights'];
        if self.verbose == True:
            print "-connect matrix re-created.";


        """ Connection Status Matrix """
        #recreate connection matrix
        self.conn_active = self.genome['connectivity'];
        if self.verbose == True:
            print "-connect status matrix  re-created.";


        """(4) Bias and its weight """
        """ Bias Matrix"""
        #create bias Matrix
        self.biasMatrix = copy.deepcopy(self.genome['bias']);
        if self.verbose == True:
            print "-bias matrix created.";


        """ Autoweight Matrix"""
        #create autoWeight Matrix
        self.autoWeightMatrix = copy.deepcopy(self.genome['biasWeights']);
        if self.verbose == True:
            print "-bias matrix created.";


        """(5) Other """
        num_hidden = netParams.nodeConfig['H'];
        
        """ Output Matrix """
        #recreate output matrix
        self.outputMatrix = numpy.zeros((activNodeCount),numpy.float32);
        if self.verbose == True:
            print "-output matrix created.";

    
        """ Context Layer """
        if(self.use_context):
            
            #get number of hidden nodes
            num_hidden_activ = self.getNumActiveHidden();
            #debug logging
            if self.debug == True:
                #logging.info("num_hidden_activ");
                print "num_hidden_activ", num_hidden_activ;
            
            #create context matrix
            self.contextMatrix = numpy.zeros((num_hidden_activ),numpy.float);
            if self.verbose == True:
                print "-context matrix created.";

            #create context weight matrix
            self.contextWeights = numpy.ones((num_hidden_activ ), numpy.float32);
            if self.debug == True:
                print "-context weights created.";            
                print "contextMat", self.contextMatrix;
                print "contextWeights", self.contextWeights;
                

        """Relational Nodes"""
        if(self.use_nodeRel):
            #create node relation output matrix
            self.nodeRelMat = numpy.zeros((self.num_rel_nodes), numpy.float32);
            #create node relation weight matrix
            self.nodeRelWeights = numpy.ones((self.num_rel_nodes,num_hidden), numpy.float32);
            
        
            if self.verbose == True:
                print "-relationals established.";

        
        #create nodes
        self.nodes = self.recreateNodes();


    def createNet(self,genome = None):
        """create network based on configuration"""

        #count nodes by configuration
        nodeCount = self.nodeConfig['I'] + self.nodeConfig['H'] + self.nodeConfig['O'];

        #create connection matrix
        if(nodeCount > 0):

            ##  NEW NETWORK WITHOUT GENOME
            """ Connection Matrix """
            #create connection matrix
            self.connMatrix = numpy.zeros((nodeCount,nodeCount), numpy.float32);
            if self.verbose == True:
                print "-connect matrix created.";

            """ Connection Status Matrix """
            #create connection matrix
            self.conn_active = numpy.zeros((nodeCount,nodeCount), numpy.float32);
            if self.verbose == True:
                print "-connect status matrix  created.";
            
            """ Output Matrix """
            #create output matrix
            self.outputMatrix = numpy.zeros((nodeCount),numpy.float32);
            if self.verbose == True:
                print "-output matrix created.";

            """ Node status Matrix """
            #create output matrix
            self.nodes_active = numpy.ones((nodeCount),numpy.float32);
            if self.verbose == True:
                print "-output matrix created.";
            
            """ Bias Matrix"""
            #create bias Matrix
            self.biasMatrix = numpy.ones((nodeCount),numpy.float32);
            if self.verbose == True:
                print "-bias matrix created.";

            """ Autoweight Matrix"""
            #create autoWeight Matrix
            self.autoWeightMatrix = numpy.zeros((nodeCount),numpy.float32);
            if self.verbose == True:
                print "-bias matrix created.";

            """ Context Layer """
            if(self.use_context):
                #create context matrix
                self.contextMatrix = numpy.zeros((self.nodeConfig['H']),numpy.float32);
                if self.verbose == True:
                    print "-context matrix created.";

                #create context weight matrix
                self.contextWeights = numpy.ones((self.nodeConfig['H']), numpy.float32);
                if self.verbose == True:
                    print "-context weights created.";

            """Relational Nodes"""
            if(self.use_nodeRel):
                #create node relation output matrix
                self.nodeRelMat = numpy.zeros((self.num_rel_nodes), numpy.float32);
                #create node relation weight matrix
                self.nodeRelWeights = numpy.ones((self.num_rel_nodes,nodeCount), numpy.float32);
                
                

                if self.verbose == True:
                    print "-relational nodes established.";
        
        
        #create nodes of the network
        self.createNodes(nodeCount);
        
            
        #finally, establish connections
        #Note: Uses an elman architecture so it might not be strictly feedforward
        if (netParams.connectionType == constants.RANDOM):
            self.connectRandomly(nodeCount);
        elif (netParams.connectionType == constans.FULLY_CONNECTED):
            self.connectFeedForward(nodeCount);
       
            
           
        #set network as ready
        self.ready = True;

    """ CREATE NETWORK - HELPER FUNCTIONS """
    def recreateNodes(self, genome = None ):
        """ recreates nodes from the network"""
        
        #logging.info('*In recreateNode ');

        if genome != None:
            self.genome = genome;

        #get number of active nodes
        activNodeCount  = self.getCountActivNodes(genome);
        
        #get node count
        nodeCount = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];

        # get transfer functions of hidden and outputs nodes
        lst_nf = copy.deepcopy(self.genome['nodeFns']);
        lst_wf = copy.deepcopy(self.genome['weightFns']);
        fnParams = copy.deepcopy(self.genome['fnParams']);
        
        
        #reverse for popping
        lst_nf.reverse();
        lst_wf.reverse();
        fnParams.reverse();

        #node id
        node_id = 0;
        
        #creates nodes and assign ids and types
        for i in xrange(nodeCount):
            
            
            #set in active nodes as inactive
            if self.isActiveNode(i) is not True:
                nd = inactive_node(i,-1);
                self.nodes.append(nd);
                #logging.info('-Inactive Node');
                continue;

            if(self.isInputNode(i,self.nodeConfig) and self.isActiveNode(i)):  #input nodes - 1
                nd = node(i,node_id,1); #input node and assign id
                nd.nodeFn = 1; #identity
                nd.weightFn = -1;
                self.autoWeightMatrix[node_id] = 1.0;
                self.outputMatrix[node_id] = 0.0;
                nd.fnParam = [];
                self.nodes.append(nd);
                node_id += 1;
                #logging.info('-Input Node');
                
            if(self.isHiddenNode(i,self.nodeConfig) and self.isActiveNode(i)):  #hidden nodes - 2
                #hidden node
                nd = node(i,node_id,2);
                #logging.info('-Hidden Node');
            
            if(self.isOutputNode(i,self.nodeConfig) and self.isActiveNode(i)):  #output nodes - 3
                #output node
                nd = node(i,node_id,3);
                #logging.info('-Output Node');
                
            if((self.isHiddenNode(i,self.nodeConfig) or self.isOutputNode(i,self.nodeConfig)) and
                self.isActiveNode(i)):  #only to hidden and output nodes
                
                #choose node function
                if self.debug or self.verbose :
                    print "node functions:", lst_nf;
                    print "weight functions:", lst_wf;
                    
                nd.nodeFn = lst_nf.pop();
                
                
                #chose weight function 
                nd.weightFn = lst_wf.pop();
                
                #value
                self.outputMatrix[node_id]= 0.0;
                
                #bias
                adjW = self.biasMatrix[node_id];
                if nd.weightFn == constants.INNER_PROD:
                    adjW = self.putInRange(adjW,self.param_limits['minWB'],
                                              self.param_limits['maxWB']);
                self.biasMatrix[node_id] = adjW;
                    
                #autoweight
                if self.debug == True:
                    print "autoweight";
                    print self.autoWeightMatrix;
                    
                    
                adjW = self.autoWeightMatrix[node_id];
                if nd.weightFn == constants.INNER_PROD:
                    adjW = self.putInRange(adjW,self.param_limits['minWB'],
                                              self.param_limits['maxWB']);
                self.autoWeightMatrix[node_id] = adjW;
                    
                #function parameters
                nd.fnParam.extend(fnParams.pop());
                
                if self.debug == True:
                    print "fnParams", nd.fnParam;
                    ##logging.debug('fnParams',nd.fnParam);
                       
                #add network       
                self.nodes.append(nd);
                
                #increament node_id
                node_id += 1;
                
        #return nodes
        return self.nodes;


    def createNodes(self, nodeCount):
        """ creates nodes for the network """
        if self.verbose == True:
            print "-creating nodes (in createNodes method)";
            
        #creates nodes and assign ids and types
        for i in xrange(nodeCount):
            
            if(self.isInputNode(i,self.nodeConfig)):  #input nodes - 1
                nd = node(i,1); #input node and assign id
                nd.nodeFn = 1; #identity
                nd.weightFn = -1;
                self.autoWeightMatrix[i] = 1.0;
                self.biasMatrix[i] = 0.0;
                self.outputMatrix[i] = 0.0;
                self.nodes_active[i] = 1;
                nd.fnParam = [];
                self.nodes.append(nd);
                
            if(self.isHiddenNode(i,self.nodeConfig)):  #hidden nodes - 2
                #hidden node
                nd = node(i,2);
                
            if(self.isOutputNode(i,self.nodeConfig)):  #output nodes - 3
                #output node
                nd = node(i,3);
                
            if(self.isHiddenNode(i,self.nodeConfig) or self.isOutputNode(i,self.nodeConfig)):  #only to hidden and output nodes  
                #choose node function  - uniform distribution
                nd.nodeFn =netParams.nodeFns[int(numpy.random.rand() * len(netParams.nodeFns) -1)];
                
                #chose weight function - uniform distribution
                nd.weightFn = netParams.weightFns[int(numpy.random.rand() * len(netParams.weightFns) -1)];
                
                #value
                self.outputMatrix[i]= 0.0;
                
                #bias
                adjW = numpy.random.rand();
                if nd.weightFn == constants.INNER_PROD:
                    adjW = self.putInRange(adjW,self.param_limits['pMin'],
                                              self.param_limits['pMax']);
                self.biasMatrix[i] = adjW;
                    
                #autoweight
                adjW = numpy.random.rand();
                if nd.weightFn == constants.INNER_PROD:
                    adjW = self.putInRange(adjW,self.param_limits['pMin'],
                                              self.param_limits['pMax']);
                self.autoWeightMatrix[i] = adjW;
                    
                #function parameters
                #IDENTITY
                if nd.nodeFn == constants.IDENTITY:
                    nd.fnParam = [];
                #SIGMOID
                if nd.nodeFn == constants.SIGMOID:
                    nd.fnParam = [self.putInRange(numpy.random.rand(),
                                                 self.param_limits['sig_minFP'],
                                                 self.param_limits['sig_maxFP'])
                                  ];
                #GAUSS
                if nd.nodeFn == constants.GAUSSIAN:
                    nd.fnParam = [self.putInRange(numpy.random.rand(),
                                                  self.param_limits['gaus_minFP1'],
                                                  self.param_limits['gaus_maxFP1']),
                                  self.putInRange(numpy.random.rand(),
                                                  self.param_limits['gaus_minFP2'],
                                                  self.param_limits['gaus_maxFP2'])
                                ];
                #TANH
                if nd.nodeFn == constants.TANH:
                    nd.fnParam = [self.putInRange(numpy.random.rand(),
                                                 self.param_limits['tanh_minFP'],
                                                 self.param_limits['tanh_maxFP'])
                                  ];

                #GAUSS II
                if nd.nodeFn == constants.GAUSSIAN_II:
                    nd.fnParam = [self.putInRange(numpy.random.rand(),
                                                  self.param_limits['gaus_minFP1'],
                                                  self.param_limits['gaus_maxFP1']),
                                  self.putInRange(numpy.random.rand(),
                                                  self.param_limits['gaus_minFP2'],
                                                  self.param_limits['gaus_maxFP2'])
                                ];
                #PROB. SIGMOID
                if nd.nodeFn == constants.PROB_SIGMOID:
                    nd.fnParam = [self.putInRange(numpy.random.rand(),
                                                 self.param_limits['sig_minFP'],
                                                 self.param_limits['sig_maxFP'])
                                  ];

                #add the node
                self.nodes.append(nd);

    def getCountActivNodes(self, genome = None ):
        """ return the number of active nodes"""
        if(genome != None):
            self.genome = genome;

        return self.genome['architecture'].count(constants.ACTIVE);
 

    def connectFeedForward(self, nodeCount):
        """ computational nodes to themselves """
        for i in xrange(nodeCount):
            for j in xrange(nodeCount):
                #connect in feed-forward network (allow to grow complexity)
                if(i != j):
                    
                    #input to hidden nodes
                    if(self.isInputNode(i,self.nodeConfig) and
                       self.isHiddenNode(j,self.nodeConfig)):
                        #set connection as active
                        self.conn_active[i][j] = 1; #active
                        #create connection
                        self.connMatrix[i][j] = numpy.random.rand();

                    #hidden to output nodes
                    if(self.isHiddenNode(i,self.nodeConfig) and
                       self.isOutputNode(j,self.nodeConfig)):
                        #set connection as active
                        self.conn_active[i][j] = 1; #active
                         #create connection with random weight
                        self.connMatrix[i][j] = numpy.random.rand();
                else:
                    #not connected
                    self.connMatrix[i][j] = 0;
                    #set connection as not active
                    self.conn_active[i][j] = 0; #not-active
                    
                    
    def connectRandomly(self, nodeCount):
        """ Connects the network in a random manner, There are rules that would govern these connections """
        
        for i in xrange(nodeCount):
            for j in xrange(nodeCount):
                #connect in feed-forward network (allow to grow complexity)
                if(i != j):
                    
                    #only input to hidden or output layer connections (there should be no in to in)
                    #INPUT TO HIDDEN NODES
                    if(self.isInputNode(j, self.nodeConfig) and
                       self.isHiddenNode(i, self.nodeConfig) and
                       netParams.conn_limits['inToMid'] == constants.YES):
                        
                        probConn = numpy.random.rand();
                        
                        if (netParams.conn_limits['probConn'] <= probConn):
                            
                            if debug_connRandom:
                                print "Probability of connection: ",probConn;
                                print " - connection established (", i,j,")";
                        
                            #set connection as active
                            self.conn_active[i][j] = 1; #active
                            #create connection
                            self.connMatrix[i][j] = numpy.random.rand();
                            
                            
                    #INPUT TO OUTPUT NODE
                    if(self.isInputNode(j, self.nodeConfig) and
                       self.isOutputNode(i, self.nodeConfig) and
                       netParams.conn_limits['inToOut'] == constants.YES):
                        
                        probConn = numpy.random.rand();
                        
                        if (netParams.conn_limits['probConn'] <= probConn):
                            
                            if debug_connRandom:
                                print "Probability of connection: ",probConn;
                                print " - connection established (", i,j,")";
                        
                            #set connection as active
                            self.conn_active[i][j] = 1; #active
                            #create connection
                            self.connMatrix[i][j] = numpy.random.rand();
                            
                    #HIDDEN TO HIDDEN NODE - LATERAL HIDDEN NODE CONNECTIONS
                    if(self.isHiddenNode(j, self.nodeConfig) and
                       self.isHiddenNode(i, self.nodeConfig) and
                       netParams.conn_limits['midToMid'] == constants.YES):
                        
                        probConn = numpy.random.rand();
                        
                        if (netParams.conn_limits['probConn'] <= probConn):
                            
                            if debug_connRandom:
                                print "Probability of connection: ",probConn;
                                print " - connection established (", i,j,")";
                        
                            #set connection as active
                            self.conn_active[i][j] = 1; #active
                            #create connection
                            self.connMatrix[i][j] = numpy.random.rand();
                            
                    #HIDDEN TO OUTPUT NODE
                    if(self.isHiddenNode(j, self.nodeConfig) and
                       self.isOutputNode(i, self.nodeConfig) and
                       netParams.conn_limits['midToOut'] == constants.YES):
                        
                        probConn = numpy.random.rand();
                        
                        if (netParams.conn_limits['probConn'] <= probConn):
                            
                            if debug_connRandom:
                                print "Probability of connection: ",probConn;
                                print " - connection established (", i,j,")";
                        
                            #set connection as active
                            self.conn_active[i][j] = 1; #active
                            #create connection
                            self.connMatrix[i][j] = numpy.random.rand();
                            
                else:
                    #not connected
                    self.connMatrix[i][j] = 0;
                    #set connection as not active
                    self.conn_active[i][j] = 0; #not-active

    """ --- END OF HELPER FUNCTIONS --- """      
                
            
    def stimulate(self,pattern):
        """stimulates the nerual network with the given pattern"""
        
        ##logging
        #logging.info('*Stimulate Network');
        
        #get pattern length
        patt_len = len(pattern);
        #response to empty pattern
        if patt_len == 0:
            return [0.0];
        #input pattern length and input nodes comparison
        if self.countActiveInputs() != patt_len:
            print "Issue: Input nodes not same as number of stimulus data";
            print "Required", patt_len, "inputs";
            
        
        #place pattern stimuli to input nodes
        self.outputMatrix[0:patt_len] = pattern;
        if self.verbose == True:
            print "*input node values: " , self.outputMatrix[0:patt_len];
            ##logging.debug('input values:',self.outputMatrix[0:patt_len]);

        #count active nodes
        num_active_nodes = self.countActiveInputs() + self.countActiveHidden() + self.countActiveOutputs();
        
        #total nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
   
        #propagate pattern through network        
        for n_item in (self.nodes):
                    
                #only active nodes
                if  self.isActiveNode(n_item.node_id) is not True:
                    #logging.info('-skipping');
                    continue;
                    
                #node ids
                source_node = -1;
                target_node = n_item.map_id;
                
                ### weight function ###
                #inner product
                if n_item.weightFn == constants.INNER_PROD:
                        
                    #calulate activation
                    summVal = (self.autoWeightMatrix[target_node] * self.outputMatrix[target_node]);
                    output = (self.connMatrix.transpose() * self.outputMatrix);
                    #perform another tranpose on output to return to original arrangments
                    output = output.transpose();
                    #calculate incoming action potential
                    for n in self.nodes:
                        
                        if self.isActiveNode(n.node_id) and n.node_id != n_item.node_id:
                            if self.conn_active[n.map_id][target_node] == constants.ACTIVE:
                                summVal = summVal + output[n.map_id][target_node];
                            #debug
                            if self.debug == True:
                                print "-Inner product [node:", target_node ,"]";
                                print output[n.map_id][target_node];
                                ##logging.debug('Inner product output', output[n][target_node]);
                    
                    #correct NaN
                    if numpy.isnan(summVal):
                        summVal = 0.0;
                         
                    #store activation
                    self.outputMatrix[target_node] = summVal + self.biasMatrix[target_node];
                        
                """ Distance functions """
                #Euclidean distance
                if n_item.weightFn == constants.EUCLID_DIST :
                    #calulate value
                    summVal = 0; # no autoweight value
                    exp= 2; #exponent
                    
                    #euclid - exp2
                    output = (pow(self.connMatrix.transpose() - self.outputMatrix, exp));
                    #perform another tranpose on output to return to original arrangments
                    output = output.transpose();
                    #calculate incoming action potential
                    for n  in self.nodes:
                        if self.isActiveNode(n.node_id) and n.node_id != n_item.node_id:
                            if self.conn_active[n.map_id][target_node] == constants.ACTIVE:
                                summVal = summVal + output[n.map_id][target_node];
                            #debug
                            if self.debug == True:
                                print "-Euclid [node:", target_node ,"]";
                                print output[n.map_id][target_node];
                                
                    #correct NaN
                    summVal = math.sqrt(summVal);
                    if math.isnan(summVal):
                        summVal = 0.0;
                    
                    
                    #store activation
                    self.outputMatrix[target_node] = summVal;
                    
                #Manhattan distance
                if n_item.weightFn == constants.MANHATTAN_DIST:
                    #calculate value
                    summVal = 0;
                    
                    #manhattan - exp 1
                    output = (self.connMatrix.transpose() - self.outputMatrix);
                    #perform another tranpose on output to return to original arrangments
                    output = output.transpose();
                    #calculate incoming action potential
                    for n  in self.nodes:
                        if self.isActiveNode(n.node_id) and n.node_id != n_item.node_id:
                            if self.conn_active[n.map_id][target_node] == constants.ACTIVE:
                                
                                #only absolute values of output
                                if output[n.map_id][target_node] < 0:
                                    output[n.map_id][target_node] = output[n.map_id][target_node] * -1;
                                
                                summVal = summVal + output[n.map_id][target_node];
                            #debug
                            if self.debug == True:
                                print "-Manhattan [node:", target_node ,"]";
                                print output[n.map_id][target_node];
                                
                    #correct NaN
                    if math.isnan(summVal):
                        summVal = 0.0;
                        
                    
                    
                    #store activation
                    self.outputMatrix[target_node] = summVal;
                
                #Maximum distance
                if n_item.weightFn == constants.MAX_DIST:
                    #calculate value
                    maxVal = 0;
                    
                    #max - exp 1
                    output = (self.connMatrix.transpose() - self.outputMatrix);
                    #perform another tranpose on output to return to original arrangments
                    output = output.transpose();
                    #calculate incoming action potential
                    for n  in self.nodes:
                        if self.isActiveNode(n.node_id) and n.node_id != n_item.node_id:
                            if self.conn_active[n.map_id][target_node] == constants.ACTIVE:
                                #pick the maximum value
                                maxVal = max(maxVal,output[n.map_id][target_node]);
                            #debug
                            if self.debug == True:
                                print "-Manhattan [node:", target_node ,"]";
                                print output[n.map_id][target_node];
                                
                    #correct NaN
                    if math.isnan(summVal):
                        summVal = 0.0;
                        
                    #store activation
                    self.outputMatrix[target_node] = maxVal;
                    
                    
                #Higher order product
                if n_item.weightFn == constants.HIGHER_ORDER_PROD :
                    #calulate value
                    prodVal = 1; # no autoweight value
                 
                    #higher order product
                    output = ((self.connMatrix.transpose() * self.outputMatrix) * self.prodConstant);
                    
                    #perform another tranpose on output to return to original arrangments
                    output = output.transpose();
                    #deactivate in active connections
                    output = output * self.conn_active;
                    
                    if self.debug == True:
                        print "-Higher order product (output) [node:", target_node ,"]";
                        print output;

                    #get action potentials (incoming)
                    inVec = output[:,target_node];

                    if self.debug == True:
                        print "-inVec:", inVec;
                    

                    #calculate incoming action potential
                    #- multiplies only non-zeros
                    for ap in inVec:
                        #if(ap == 0.0):
                        #    ap = 0.00001;
                        prodVal = prodVal *  ap;
                        #debug
                        if self.debug == True:
                            print "-Higher-Order prod.(accm. potential) [node:", target_node ,"]";
                            print prodVal;

                    #correction (when all values deactivated)
                    #-This is prevent giving out a false output of (1).
                    #if (sum(inVec) == 0):
                    #    prodVal = 0;
                    
                    #NaN correction
                    if math.isnan(prodVal):
                        prodVal = 0.0;
                        
                    #store activation
                    self.outputMatrix[target_node] = prodVal;
                    #debug
                    if self.debug == True:
                        print "-Higher order product (node output) [node:", target_node ,"]";
                        print self.outputMatrix[target_node];


                #Higher-order subtractive 
                if n_item.weightFn == constants.HIGHER_ORDER_SUB :
                    #init sum
                    summVal = 0;

                    #get first active connection from an active node
                    firstActivConn = self.getFirstActiv(target_node);
                    
                    #debug
                    if self.debug or self.verbose:
                        print ">>OUTPUT MATRIX";
                        print self.outputMatrix;

                    #higher order subtractive
                    output = ((self.connMatrix.transpose() * self.outputMatrix));
                    #perform another tranpose on output to return to original arrangments
                    output = output.transpose();
                    if self.debug == True:
                        print "-Higher order subtractive [node:", target_node ,"]";
                        print output;

                    
                    #if there is a connection 
                    if firstActivConn !=  -1:
                        
                        #get the value of the node
                        firstVal = output[firstActivConn][target_node];

                        #subtract from other inputs
                        for n in xrange(len(self.nodes) - 1):
                            
                            #skip inactive nodes
                            if self.nodes_active[n] == constants.INACTIVE:
                                continue;
                            
                            if self.conn_active[n][target_node] == constants.ACTIVE \
                            and self.nodes_active[n] == constants.ACTIVE:
                                if firstActivConn != n :
                                    nVal = output[n][target_node];
                                    summVal = summVal + (firstVal - nVal);

                                #debug
                                if self.debug == True:
                                    print "-Higher-Order prod.[node:", target_node ,"]";
                                    print summVal;
                            
                    #correct NaN
                    if math.isnan(summVal):
                        summVal = 0.0;
                        
                    #store activation
                    self.outputMatrix[target_node] = summVal;
                    #debug
                    if self.debug == True:
                        print "-Higher order substractive (node output) [node:", target_node ,"]";
                        print self.outputMatrix[target_node];

                        
                        
                #standard deviation
                if n_item.weightFn == constants.STD_DEV :

                     #calculate inner product 
                     output = (self.connMatrix.transpose() * self.outputMatrix);


                     #perform another tranpose on output to return to original arrangments
                     output = output.transpose();

                     #deactivate inactive connections
                     output = output * self.conn_active;

                     #get inVec with swtiched off connections
                     inVec = output[:,target_node];
                    
                     #calculate std
                     val = numpy.std(inVec);

                     #debug
                     if self.debug == True:
                         print "-Standard Dev [node:", target_node ,"]";
                         print val;

                     #correct std. dev values that are not numbers
                     if numpy.isnan(val):
                         val = 0.0;

                     #sore activation
                     self.outputMatrix[target_node] = val;

                     #debug
                     if self.debug == True:
                        print "-Standard Dev. (node output) [node:", target_node ,"]";
                        print self.outputMatrix[target_node];
                        
                        
                #mean
                if n_item.weightFn == constants.MEAN :
                     #calculate inner product 
                     output = ((self.connMatrix.transpose() * self.outputMatrix));
                     
                     
                     #perform another tranpose on output to return to original arrangments
                     output = output.transpose();
                    
                     #deactivate inactive connections
                     output = output * self.conn_active;
                    
                     
                     #get input action potentials that are active
                     inVec = output[:,target_node];
                     

                     #mean
                     val = inVec.mean();
                     # correct the value
                     if len(inVec) == 0 or numpy.isnan(val):
                         val = 0.0;

                     #store action potential
                     self.outputMatrix[target_node] = val;

                     #debug
                     if self.debug == True:
                        print "-Min (node output) [node:", target_node ,"]";
                        print self.outputMatrix[target_node];
    

                #min
                if n_item.weightFn == constants.MIN :
                     #calculate inner product 
                     output = ((self.connMatrix.transpose() * self.outputMatrix));
                     
                     
                     #perform another tranpose on output to return to original arrangments
                     output = output.transpose();
                    
                     #deactivate inactive connections
                     output = output * self.conn_active;
                    
                     
                     #get input action potentials that are active
                     inVec = output[:,target_node];
                     

                     #minimum
                     val = inVec.min();
                     # correct the value
                     if len(inVec) == 0 or numpy.isnan(val):
                         val = 0.0;

                     #store action potential
                     self.outputMatrix[target_node] = val;

                     #debug
                     if self.debug == True:
                        print "-Min (node output) [node:", target_node ,"]";
                        print self.outputMatrix[target_node];

                #max
                if n_item.weightFn == constants.MAX :
                     #calculate inner product 
                     output = ((self.connMatrix.transpose() * self.outputMatrix));
                     
                     
                     #perform another tranpose on output to return to original arrangments
                     output = output.transpose();

                     #deactivate inactive connections
                     output = output * self.conn_active;
                     
                     #get input action potentials that are active
                     inVec = output[:,target_node];
                    

                     #maximum
                     val = inVec.max();
                     # correct the value
                     if len(inVec) == 0 or numpy.isnan(val):
                         val = 0.0;
                        
                    
                     #debug
                     if self.debug == True:
                        print "-max [node:", target_node ,"]";
                        print "input potentials :",inVec;
                        print "value:", val;

                     #store action potential
                     self.outputMatrix[target_node] = val;
                    

                """ Contex layer - injection of values"""
                ### Add context layer information to hidden nodes ###
                if(self.use_context == True):
                    if(self.isActiveHiddenNode(n_item.node_id)):
                        
                        #start and end of hidden nodes
                        hid_start = netParams.nodeConfig['I'];
                        hid_end = netParams.nodeConfig['I'] + self.countActiveHidden();
                        num_hid = hid_end - hid_start;
                        
                        #get current value after combination
                        curr_out = self.outputMatrix[target_node];
                        #multiply by weights
                        if self.debug == True:
                            print self.contextMatrix;
                            print self.contextWeights;
                        #mutiply
                        contextMat = self.contextMatrix * self.contextWeights;
                        #reverse for popping
                        if self.debug == True:
                            print ">>output:",self.outputMatrix[hid_start:hid_end];
                            print ">>context:",contextMat;
                            
                        
                        #inject previous value
                        injectValues = self.outputMatrix[hid_start:hid_end] + contextMat;
                         
                        if math.isnan(sum(injectValues)):
                            print "-injection skipped";
                            self.outputMatrix[hid_start:hid_end] = self.outputMatrix[hid_start:hid_end];
                        
                        #debug
                        if(self.debug==True):
                            print "-Elman Context : Information injection"
                            print "current_action_potential:", curr_out;
                            print "new_action_potential:",self.outputMatrix[target_node];

                
                """ NODE FUNCTIONS -
                    * should you want to add a node function,
                    add a it to this list of conditional blocks
                """
                #IDENTITY
                if n_item.nodeFn == constants.IDENTITY :
                    #apply activation function
                    val =  self.outputMatrix[target_node];
                    
                    if (self.problem_type == constants.REGRESSION and\
                        (n_item.type == constants.HIDDEN_NODE or n_item.type == constants.OUTPUT_NODE)):
                        val = float(val *n_item.fnParam[0]); 
                       

                    elif(self.problem_type == constants.CLASSIFICATION and \
                         (n_item.type== constants.OUTPUT_NODE or n_item.type == constants.HIDDEN_NODE )):
                        smthQ = n_item.fnParam[0]; #smoothing quoef.
                        val = float(self.identity_fn(val));
                

                    #debug
                    if self.debug == True:
                        print "-identity output [node:", target_node ,"]";
                        print val;
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;
                
                    #store
                    self.outputMatrix[target_node] = val;
                    
                    

                #SIGMOID FUNCTION
                if n_item.nodeFn == constants.SIGMOID :
                    #apply activation function
                    val = self.sigmoid_fn(self.outputMatrix[target_node],
                                          n_item.fnParam[0],
                                          n_item.fnParam[1]);
                    #debug
                    if self.debug == True:
                        print "-simoid output [node:", target_node ,"]";
                        print val;
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;
                        
                    #store
                    self.outputMatrix[target_node] = val;

                #GAUSSIAN
                if n_item.nodeFn == constants.GAUSSIAN :
 
                    #apply activation function
                    val = self.gaussian_fn(self.outputMatrix[target_node],
                                           n_item.fnParam[0]);
                   
                    
                    #debug
                    if self.debug == True:
                        print "-gaussian output [node:", target_node ,"]";
                        print val;
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;

                    #store
                    self.outputMatrix[target_node] = val;
                    
                
                #THIN-PLATE-SPLINE
                if n_item.nodeFn == constants.THIN_PLATE_SPLINE :
 
                    #apply activation function
                    val = self.thin_plate_spline_fn(self.outputMatrix[target_node],
                                                    n_item.fnParam[0]);
                   
                    
                    #debug
                    if self.debug == True:
                        print "-thin_plate_spline output [node:", target_node ,"]";
                        print val;
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;

                    #store
                    self.outputMatrix[target_node] = val;

                #TANH
                if n_item.nodeFn == constants.TANH :

                    #apply activation function
                    val = self.tanh_fn(self.outputMatrix[target_node],
                                        n_item.fnParam[0]);

                    #debug
                    if self.debug == True:
                        print "-tanh output [node:", target_node ,"]";
                        print val;
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;


                    #sore
                    self.outputMatrix[target_node] = val;
                    
                    
                #ARC_TAN
                if n_item.nodeFn == constants.ARC_TAN:
                    
                    #apply activation function
                    val = self.arc_tanh_fn(self.outputMatrix[target_node],
                                        n_item.fnParam[0]);
                    
                    #debug
                    if self.debug:
                        print "-arch tanh [node", target_node,"]";
                        print val;
                        
                    #handle NaN values
                    if math.isnan(val):
                        val = 0.0;
                        
                    #store
                    self.outputMatrix[target_node] = val;
                        
                    

                #GAUSSIAN II
                if n_item.nodeFn == constants.GAUSSIAN_II :
                    
            
                    #apply activation function
                    val = self.gaussian_ii_fn(self.outputMatrix[target_node],
                                              n_item.fnParam[0],
                                              n_item.fnParam[1]);

                    #debug
                    if self.debug == True:
                        print "-gaussian ii output [node:", target_node ,"]";
                        print val;
                        
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;

                    #store
                    self.outputMatrix[target_node] = val;
                
                #PROBABLISTIC SIGMOID
                if n_item.nodeFn == constants.PROB_SIGMOID :
                    
                    #apply activation function
                    val = self.prob_sigmoid_fn(self.outputMatrix[target_node],
                                               n_item.fnParam[0],
                                               n_item.fnParam[1]);
                    
                    #debug
                    if self.debug == True:
                        print "--prob sigmoid ouput [node:", target_node ,"]";
                        print val;
                        
                    #handle NaN values
                    if numpy.isnan(val):
                        val = 0.0;
                        
                     #store
                    self.outputMatrix[target_node] = val;
                                        
        # COPY TO CONTEXT LAYER
        if(self.use_context):
            self.updateContextLayer(self.outputMatrix);

        #add to output history
        outcopy = copy.deepcopy(self.outputMatrix.tolist());
        self.outputHistory.append(outcopy);

        #get final network output
        return self.getOutput();
    
    def evaluate(self, pattern , net = None, return_pattern = False,use_gauss_noise = False):
        """ evaluates how a network does on a pettern """
        MSE = 0.0;
       
        #input/output cache
        inPattern = [];
        
        
        #pattern as arg.
        if net == None:
            
            #get inputs and outputs
            inputs = pattern['IN'];
            outputs = pattern['OUT'];
            
            #if use random sampling training
            if self.use_random_sample_wreplacement:
                
                if self.debug or self.verbose or True:
                    print ">>using random sampling"
                #zip
                Data = zip(inputs,outputs);
                #shuffle
                random.shuffle(Data);
                #unzip
                inputs,outputs = zip(*Data);
                #convert back to lists
                inputs = list(inputs);
                outputs = list(outputs);
            
            #count number of patterns
            pattern_len = len(inputs);
            
            
            #stores output pattern of solution
            if return_pattern:
                out_pattern = numpy.zeros((pattern_len),numpy.float32);

            #errors array and expected output in array form
            errors = [];
            exp_outs = numpy.array(outputs[:]);
           
            
            if self.debug == True:
                print "exp outputs",exp_outs;

            #vars
            err = -1;
            
            for pat_i in xrange(pattern_len):
                
                #skip empty patterns
                if inputs[pat_i] == []:
                    continue;
                
                #copy inputs 
                inPattern = list(inputs[pat_i]);
                
                
                #pattern debug/verbose
                if self.debug or self.verbose:
                    print ">>input:", inputs[pat_i];
                    print ">>expected output:", outputs[pat_i];
                    
                
                #stimulate Network
                #-check if gaussian noise is to be added
                if use_gauss_noise == True:
                    mu, sigma = 0, 0.1 # mean and standard deviation
                    noise = numpy.random.normal(mu, sigma, len(inputs[pat_i]));
                    inPattern  = list(inputs[pat_i] + noise);
                    
                    #debug/verbose
                    if self.debug or self.verbose :
                        print "#Gaussian noise injection:"
                        print ">>original inputs:", inputs[pat_i];
                        print ">>distorted inputs:", inPattern;
                    
                
                #-check if non-linear transfrom is to be used on inputs.
                if self.use_nonlinear_trans == True:
                    pass;
                
                #stimulate with pattern
                out = self.stimulate(inPattern);
                
                #output debug/verbose
                if self.debug or self.verbose :
                    print ">>", out;
                
                #store output pattern
                if return_pattern:
                    out_pattern[pat_i] = max(out);
                
                #calculate error
                err = (pow((exp_outs[pat_i]) - (max(out)), 2));
                
                #append to err list
                errors.append(err);

        #find the MSE
        MSE = (sum(errors)/float(pattern_len));
        MSE = (sum(errors))
        
        if self.debug == True:
            print "-MSE Error";
            print MSE;
            print "-MSE (%)";
            print MSE * 100, "%";

        if return_pattern:
            return MSE,out_pattern;
        else:
            return MSE;
            
        
        
    def updateContextLayer(self, outMat):
        """ saves the current outputs of the hidden nodes """
        
        #clear old matrix
        self.contextMatrix = [];

        for n in self.nodes:
            
            if (self.isActiveHiddenNode(n.node_id)):
                #add value
                self.contextMatrix.append(self.outputMatrix[n.map_id]);
                

    def putInRange(self,geneVal, minVal, maxVal):
        #puts value within range
        if geneVal < minVal:
            geneVal = minVal;
        if geneVal > maxVal:
            geneVal = maxVal;
            
        return geneVal;


    def printNodesInfo(self):
        #prints the information of the nodes
            
        print "\n +++++++ Nodes +++++++";
        for nd in self.nodes:
            
            if self.isActiveNode(nd.node_id):
                #print node information
                print "\n#NODE#";
                print "\n-node id: ", nd.node_id;
                print "\n-map id: ", nd.map_id;
                print "\n-type: ",  nd.type;
                print "\n-active: ", self.nodes_active[nd.node_id];
                print "\n-weightFn: ",  nd.weightFn;
                print "\n-nodeFn: ",  nd.nodeFn;
                print "\n-autoWeight: ",  self.autoWeightMatrix[nd.map_id];
                print "\n-bias: ",  self.biasMatrix[nd.map_id];
                print "\n-value: ", self.outputMatrix[nd.map_id];
                print "\n-fnParam: ", nd.fnParam;
                print "\n";
            else:
                #print node information
                print "\n#NODE#";
                print "\n-node id: ", nd.node_id;
                print "\n-type: ",  nd.type;
                print "\n-active: ", self.nodes_active[nd.node_id];
                print "\n-weightFn: ",  nd.weightFn;
                print "\n-nodeFn: ",  nd.nodeFn;
                print "\n-autoWeight: ",[];
                print "\n-bias: ",  [];
                print "\n-value: ", 0.0;
                print "\n-fnParam: ", nd.fnParam;
                print "\n";


        print "\n +++++++++++++++++++++";

    """ Helper functions"""
    
        
    def isInputNode(self,indx, node_config):
        #checks if the node is an input node
        
        if indx < node_config['I']:
            return True;

    def isHiddenNode(self,indx,node_config):
        #checks if the node is an hidden node
        
        if indx >= node_config['I']  and indx < (node_config['I'] + node_config['H']):
            return True;
        
    def isOutputNode(self,indx,node_config):
        #checks if the node is an hidden node
        
        if ((indx >= node_config['I'] + node_config['H'] )  and
            indx < (node_config['I'] + node_config['H'] + node_config['O'])):
                return True;


    def isActiveNode(self, indx):
        #checks if the node is an active input node
        if self.nodes_active[indx] == constants.ACTIVE:
            return True;



    def countActiveInputs(self):
        #returns the count of active input nodes
        count = 0;
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        for x in xrange(num_nodes):
            if(self.isInputNode(x, netParams.nodeConfig) and self.nodes_active[x]):
                count += 1;
                
        return count;
        

    def countActiveHidden(self):
        #returns the count of active hidden nodes
        count = 0;
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        for x in xrange(num_nodes):
            if(self.isHiddenNode(x,  netParams.nodeConfig) and self.nodes_active[x]):
                count += 1;

        return count;

    def countActiveOutputs(self):
        #returns the count of active output nodes
        count = 0;
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        for x in xrange(num_nodes):
            if(self.isOutputNode(x,  netParams.nodeConfig) and self.nodes_active[x]):
                count += 1;

        return count;

    def isActiveNode(self,x):
        #returns true if the node is active

        if self.nodes_active[x]:
            return True;
        else:
            return False;
        
        
    def isActiveInputNode(self,x):
        """ returns true if its an active input node """
        
        if self.isActiveNode(x) and self.isInputNode(x,netParams.nodeConfig):
            return True;
        else:
            return False;
    
    def isActiveHiddenNode(self,x):
        """ returns true if its an active hidden node """
        
        if self.isActiveNode(x) and self.isHiddenNode(x,netParams.nodeConfig):
            return True;
        else:
            return False;
        
    def isActiveOutputNode(self,x):
        """ returns true if its an active output node """
        
        if self.isActiveNode(x) and self.isOutputNode(x,netParams.nodeConfig):
            return True;
        else:
            return False;
        

    def getNumActiveHidden(self ):
        """returns active hidden nodes (count) """
        count = 0;
        for x in xrange(len(self.nodes_active)):
            #count only if its active
            if self.isHiddenNode(x,self.nodeConfig) and self.isActiveNode(x):
                count += 1;

        return count;
                
                
    def getOutput(self):
        """ returns the output of the network """
        output = [];
        
        for n in self.nodes:
            #output node is always active
            if(self.isOutputNode(n.node_id,self.nodeConfig)) :
                out = self.outputMatrix[n.map_id];
                output.append(out);
            
        return output;


    def getNumActivConn(self,indx,connMatrix):
        #checks the number of connections to the node
        numConn = 0;
        for n in xrange(len(connMatrix)):
            if conn_active[n][indx] == constants.ACTIVE:
                numConn = numConn + 1;

        return numConn;


    def getFirstActiv(self, indx):
        #returns the first active node connected to the current node
        firstActiv = -1;
        
        for n in xrange(len(self.connMatrix)):
            if self.connMatrix[n][indx] == constants.ACTIVE:
               if(self.isActiveInputNode(n) or self.isActiveHiddenNode(n) or self.isActiveOutputNode(n)):
                   firstActiv = n;
               

        return firstActiv;


    """ Activation Functions """
    def identity_fn(self,x):
        #returns normalised identity of the given function
        smooth_param = 100;
        dSquared = math.pow(x,2);
        return  numpy.float(x)/((math.sqrt(dSquared + smooth_param)));

    
    def sigmoid_fn(self, x , steepness,c):
        #sigmoid activation function
        return (numpy.float(c))/(1 + (numpy.exp(-x * steepness) ));
        
        
    def gaussian_fn(self, x, width):
        #gaussian activation function
        #divide by zero fix
        if width == 0.0:
            width = 0.1;
        return math.exp(-(pow(x,2))/(width));
    
    def thin_plate_spline_fn(self, x, width):
        #thin plate spline
        exp = 2;
        return math.pow(x*width,exp) * math.log(x*width);
    

    def tanh_fn(self, x, steepness):
        #tanh activation function
        return numpy.tanh(x * steepness);
    
    def arc_tanh_fn(self, x, steepness):
        #arc tanh output function
        return math.atan(x * steepness);

    def gaussian_ii_fn(self, x, width, cut_off):
        #gaussian II activation
        val = self.gaussian_fn(x,width);
        #cut-off at threshold
        if val >= cut_off:
            val = 1;
        #return value
        return val;
    
    def prob_sigmoid_fn(self,x,steepness,c):
        #Porbablistic sigmoid activation function
        """ A node function that is probablistic in nature;
        the output of a sigmoid becomes the probability of output,
        which if a random number is below the probability fires the identity value of the function"""
        val = x;
        prob = self.sigmoid_fn(x,steepness,c);
        rn = numpy.random.rand();
        if rn <= prob:
            return (x);
        else:
            return (0.0);
        
