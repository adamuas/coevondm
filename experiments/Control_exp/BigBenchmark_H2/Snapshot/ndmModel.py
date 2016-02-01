


"""
ndmModel class

The NDM model class selects its components from the co-evolving components necessary for building a neural diversity
 machine from the pyNDMOptim. It then builds a ndm model based on the models bias.

Its supposed to be:
- An agent that searches the hypothesis space, thus has to be sufficiently diverse from each other and cover a large
space of the hypothesis space

- it has to enforce its own bias (architectural properites, etc) and build accordingly from the pieces being coevolved.
"""

debug = False;
import math;
import numpy as np;
import copy;
import random;

import node;
import activation_fn;
import output_fn;


#NODE
# 1 -FN param1
# 2 -FN Param2
# 3 -FN param3
# 4 -Bias
# 5 -Bias Weight
FN_PARAM = 3;
BIAS = 3;
BIAS_WEIGHT = 4;

#Model
#1 - Lateral connection
#2 - Context layer
#3 - Gaussian noise



#TODO : Seperate the weight functions in a seperate file (DONE)



""" CHANGES November 20th 2014"""
"""

- changed the deep copy to np.array()

- change lists to np.arrays


"""
#TODOL - Implement
#TODO: - Implemented a seperated mse fuction to enable map evaluation of the array of outptus and expected_outputs
#TODO: - benchmark to yield generator instead of send all the dataset


class ndmModel:

    #TODO: MODELS SHOULD REPRESENT DIFFERENT BIASES {PROPERTIES OF MODEL CONFIGUATIONS}
    #TODO: BIAS OF MODELS SHOULD JUST BE RANDOM BUT FIXED THRU EVOLUTION
    #TODO: Implement GP for transfer functions complexifications/mutation
    #Notes: Transfer functions complexification is likely to work better with this model
    #TODO: Prod Constant should be ranged (Consider checking how product constant changes between solutions)
    def __init__(self, numI, numH, numO, components,
                 noIntraLayerConnHH = False,
                 noContextLayer = False,
                 noNoise = False):
        """
         Initialisation

         Note - Active attribute deprecated : the active attributes for nodes has been deprecated, this is because the
         model selects to adopt the model of
        """

        self.model_id = components['model'].id;

        #bias (properties)
        self.numI = numI;
        self.numH = numH;
        self.numO = numO;
        self.inputsNodes = []; #to be created
        self.hiddenNodes = components['hidden_nodes'];
        self.outputNodes = components['out_nodes'];
        self.prodConstant = components['model'].prodConstant;
        #Lateral connections between hidden units
        if noIntraLayerConnHH:  #TODO : update this to repo (3/7/14) - [Not Done ]
            #force
            self.lateral_conn = False;
        else:
            #use genetic parameter
            self.lateral_conn = components['model'][0];
        #Context layer
        if noContextLayer:  #TODO : update this to repo (3/7/14) - [Not Done ]
            self.use_context_layer = False;
        else:
            self.use_context_layer = components['model'][1];

        #Noise inputs
        if noNoise:  #TODO : update this to repo (3/7/14) - [Not Done ]
            self.use_gauss_noise =  False
        else:
            self.use_gauss_noise = components['model'][2];

        #connections
        self.connActive_IH = components['connActive_IH'];
        self.connActive_HH = components['connActive_HH'];
        self.connActive_HO = components['connActive_HO'];
        self.connWeight_IH = components['connWeights_IH'];
        self.connWeight_HH = components['connWeights_HH'];
        self.connWeight_HO = components['connWeights_HO'];

        #other
        self.outputMatrix = np.zeros((self.numI+self.numH+self.numO), np.float32);
        self.contexLayer = np.zeros((self.numH),np.float32);




    def flush(self):
        """
        flushes remenant data from the network - very important for consitent results
        """
        self.outputMatrix.fill(0.0);
        self.contexLayer.fill(0.0);

    def stimulate(self, pattern_i):
        """
         Stimulates the model with a given pattern
        """

        #clear output matrix
        self.outputMatrix.fill(0.0);


        #Set the values for the input nodes
        if self.use_gauss_noise == True:
                mu, sigma = 0, 0.09 # mean and standard deviation
                noise = np.random.normal(mu, sigma, [self.numI]);
                self.outputMatrix[:self.numI] = pattern_i[:];
                self.outputMatrix[:self.numI] += noise;
                #debug/verbose
                if debug :
                    print "#Gaussian noise injection:"
                    print ">>original inputs:", pattern_i;
                    print ">>distorted inputs:", self.outputMatrix[:self.numI];
        else:
            self.outputMatrix[:self.numI] = pattern_i[:];


        #Caculate the weights
        weights = np.multiply(self.connActive_IH , self.connWeight_IH);

        if debug:
             print "connActiveIH", self.connActive_IH;
             print "connWeightIH", self.connWeight_IH;
             print ">>>" , weights;

        #########################PROPAGATE (INPUT TO HIDDEN)################################################
        if debug:
            print ">>> INPUT TO HIDDEN";
            print "outputMatrix:", self.outputMatrix;
        for ni,node in enumerate(self.hiddenNodes):
            #get weights for node
            node_W = weights[:,ni];

            #get the inputs for the node
            node_I = self.outputMatrix[:self.numI];
            connsActive = self.connActive_IH[:,ni];
            #ignore stimulation from inactive connections
            np.multiply(node_I,connsActive, node_I)

            #get node fnparams
            fnParams = node[:FN_PARAM];
            bias = node[BIAS];
            biasWeight = node[BIAS_WEIGHT];

            #apply the activation function
            activation = apply(node.activation_fn, [node_I, node_W]);

            if debug:
                print "node_id", node.id;
                print "I",node_I;
                print "W",node_W;
                print "FnParams", fnParams;
                print "bias", bias;
                print "biasWeight",biasWeight;
                print ">>activation:", activation;

            #apply the output function
            out = apply(node.output_fn, [activation, fnParams[0],fnParams[1], fnParams[2]]);
            out += bias * biasWeight;
            if debug:
                print ">>OUT:", out;

            #set the output of the node
            i = ni + self.numI;
            if self.use_context_layer:
                self.outputMatrix[i] = out + self.contexLayer[ni];

                if debug:
                    print "node output:", out;
                    print "context layer:", self.contexLayer[ni];

            elif not self.use_context_layer:
                self.outputMatrix[i] = out;
        ########## PROPAGATE (HIDDEN TO HIDDEN ) #############################################################
        if self.lateral_conn:
            #Check if lateral connection should be used i.e. hidden to hidden connections

            if debug:
                print ">>> HIDDEN TO HIDDEN";
                print "outputMatrix:", self.outputMatrix;

            #Caculate the weights
            weights_hh = np.multiply(self.connActive_HH , self.connWeight_HH);
            for hi,hid_node in enumerate(self.hiddenNodes):

                #get connectivity for that node
                connsActive = self.connActive_HH[:,hi];
                #get the inputs for the node
                node_I = self.outputMatrix[self.numI:self.numI+self.numH];
                #ignore activations from inactive connections
                np.multiply(node_I, connsActive,node_I);

                #get weights for node
                node_W = weights_hh[:,ni];


                #REMOVE SELF CONNECTIONS
                if debug:
                    print "Remove Self Conn:";
                    print "W_old:", node_W;
                    print "I_old:", node_I;
                #remove self connections
                w1 = node_W[:ni];
                w2 = node_W[ni+1:];
                node_W = np.hstack((w1,w2));
                i1 = node_I[:ni];
                i2 = node_I[ni+1:];
                node_I = np.hstack((i1,i2));


                #get node fnparams
                fnParams = hid_node[:FN_PARAM];
                bias = hid_node[BIAS];
                biasWeight = hid_node[BIAS_WEIGHT]

                #apply the activation function
                activation = apply(hid_node.activation_fn, [node_I, node_W]);

                if debug:
                    print "node_id", node.id;
                    print "I",node_I;
                    print "W",node_W;
                    print "FnParams", fnParams;
                    print "bias", bias;
                    print "biasWeight",biasWeight;
                    print ">>activation:", activation;

                #TODO: Something to consider, should the next nodes use the updated output of the node
                # #apply the output function
                # out = apply(node.output_fn, [activation, fnParams[0],fnParams[1], fnParams[2]]);
                # out += bias * biasWeight;

                #set the output of the node
                i = ni + self.numI;
                self.outputMatrix[i] += activation;

                if debug:
                    print ">>OUT:", out;

        ############## CONTEXT LAYER ####################################################
        #SAVE THE OUTPUTS OF THE HIDDEN UNITS IN THE CONTEXT LAYER
        if self.use_context_layer:
            self.contexLayer[:] = copy.deepcopy(self.outputMatrix[self.numI:self.numI+self.numH][:]);

            if debug:
                print "Context Layer:", self.contexLayer;

        ########## PROPAGATE (HIDDEN TO OUTPUT) ########################################

        if debug:
            print ">>> HIDDEN TO OUTPUT";
            print "outputMatrix:", self.outputMatrix;

        #Caculate the weights
        weights_ho = np.multiply(self.connActive_HO , self.connWeight_HO);

        if debug:
             print "connActiveIH", self.connActive_HO;
             print "connWeightIH", self.connWeight_HO;
             print ">>>" , weights_ho;

        for oi,out_node in enumerate(self.outputNodes):
            #get weights for node
            node_W = weights_ho[:,oi];
            #get the inputs for the node
            node_I = self.outputMatrix[self.numI:self.numI+self.numH];
            connsActive = self.connActive_HO[:,oi];
            #ignore stimulation from inactive connections
            np.multiply(node_I,connsActive, node_I);

            #get node fnparams
            fnParams = out_node[:FN_PARAM];
            bias = out_node[BIAS];
            biasWeight = out_node[BIAS_WEIGHT]


            #apply the activation function
            activation = apply(node.activation_fn, [node_I, node_W]);

            if debug:
                print "node_id", out_node.id;
                print "I",node_I;
                print "W",node_W;
                print "FnParams", fnParams;
                print "bias", bias;
                print "biasWeight",biasWeight;
                print ">>activation:", activation;

            #apply the output function
            out = apply(out_node.output_fn, [activation, fnParams[0],fnParams[1], fnParams[2]]);
            out += bias * biasWeight;
            if debug:
                print ">>OUT:", out;

            #set the output of the node
            i = oi + self.numI + self.numH;
            self.outputMatrix[i] = out;

            if debug:
                print "outputMatrix:", self.outputMatrix;


        #get network output
        net_output  = (self.outputMatrix[self.numI+self.numH:]);


        #return
        return net_output;

    def evaluate(self, dataset, verbose = False):
        """
        Evaluates the neural network on a given dataset
        """

        #clear network
        self.flush();

        #get inputs and outputs
        inputs = dataset['IN'];
        outputs = dataset['OUT'];

        #count number of patterns
        pattern_len = len(inputs);

        #errors array and expected output in array form
        errors = np.zeros(pattern_len, np.float64);
        exp_outs = np.array(outputs[:]);

        err = -1;

        for pat_i in xrange(pattern_len):

            #skip empty patterns
            if inputs[pat_i] == []:
                continue;

            #copy inputs
            inPattern = list(inputs[pat_i]);


            #pattern debug/verbose
            if debug :
                print ">>input:", inputs[pat_i];
                print ">>expected output:", outputs[pat_i];


            #stimulate with pattern
            out = self.stimulate(inPattern);

            #calculate error
            err = np.power((outputs[pat_i]) - (out),2);

            #output debug/verbose
            if debug or verbose:
                print ">>input", inputs[pat_i];
                print ">>model out:", out;
                print ">>target out:", exp_outs[pat_i];
                print ">>error:", err;

            #append to err list
            if len(err) > 1:
                err = sum(err);

            errors[pat_i]= err;


        #find the MSE
        mse = errors.mean();

        if debug == True:
            print "-MSE Error", mse;
            print "-MSE (%)", mse * 100, "%";


        return mse;

    # def squred_error(self,outputs, targets ):
    #     """
    #     Evaluates the squared error for a given pattern
    #     """
    #
    #     return  np.power(targets - outputs,2);





    # def evaluate(self, dataset, verbose = False):
    #     """
    #     An optimized implementation of the evaluation function
    #     """
    #
    #     #clear network
    #     self.flush();
    #
    #     #get inputs and outputs
    #     inputs = dataset['IN'];
    #     outputs = dataset['OUT'];
    #
    #     pattern_len = len(inputs);
    #     output_len = len(outputs[0]);
    #
    #     #expected output in array form
    #     exp_outs = np.array(outputs[:]);
    #
    #
    #     #stimulate
    #     model_outputs = map(self.stimulate, inputs);
    #
    #     #calculate the squared error
    #     errors = map(self.squred_error, model_outputs, exp_outs);
    #
    #     #find the MSE
    #     mse = sum(errors)/float(pattern_len);
    #
    #     if debug == True:
    #         print "-MSE Error", mse;
    #         print "-MSE (%)", mse * 100, "%";
    #
    #
    #     return mse;

    def getLikelihoodMatrices_FO2a(self):
        """
        Returns the likelihood matrices of first order
        :return:
        """

        weightFns = len(activation_fn.fn_indices);
        nodeFns = len(output_fn.fn_indices);

        #create transfer function combination

        transferFns_HL = np.zeros((weightFns,nodeFns),np.float32);
        transferFns_OL = np.zeros((weightFns,nodeFns),np.float32);

        #Caculate the weights
        for h_j,hid_node_j in enumerate(self.hiddenNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            transferFns_HL[act_fn_j][out_fn_j]+=1;

        for o_j, out_node_j in enumerate(self.outputNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            transferFns_OL[act_fn_j][out_fn_j]+=1;




        return {'transferFns_HL': transferFns_HL, 'transferFnsOL': transferFns_OL}

    def getLikelihoodMatrices_FO1(self):
        """
        Returns the likelihood matrices of first order
        :return:
        """

        weightFns = len(activation_fn.fn_indices);
        nodeFns = len(output_fn.fn_indices);

        #create transfer function combination

        weightFns_HL = np.zeros((weightFns),np.float32);
        weightFns_OL = np.zeros((weightFns),np.float32);
        nodeFns_HL = np.zeros((nodeFns),np.float32);
        nodeFns_OL = np.zeros((nodeFns),np.float32);

        #Caculate the weights
        for h_j,hid_node_j in enumerate(self.hiddenNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            weightFns_HL[act_fn_j]+=1;
            nodeFns_HL[out_fn_j]+=1;

        for o_j, out_node_j in enumerate(self.outputNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            weightFns_OL[act_fn_j]+=1;
            nodeFns_OL[out_fn_j]+=1;


        return {'weightFns_HL': weightFns_HL, 'nodeFns_HL':nodeFns_HL, 'weightFns_OL':weightFns_OL, 'nodeFns_OL': nodeFns_OL}



    def getLikelihoodMatrices_FO2b(self):
        """
        Returns the likelihood matrices of first order
        :return:
        """

        weightFns = len(activation_fn.fn_indices);
        nodeFns = len(output_fn.fn_indices);

        #create transfer function combination
        #Unlike the first method, this method gathers staticstics on a specific node position
        transferFn_HL_N = dict();
        transferFn_OL_N = dict();

        for i in self.hiddenNodes:
            transferFn_HL_N[str(i)] = np.zeros((weightFns,nodeFns),np.float32);

        for i in self.hiddenNodes:
            transferFn_OL_N[str(i)] = np.zeros((weightFns,nodeFns), np.float32);


        #Caculate the weights
        for h_j,hid_node_j in enumerate(self.hiddenNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            transferFn_HL_N[h_j][act_fn_j][out_fn_j]+=1;

        for o_j, out_node_j in enumerate(self.outputNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            transferFn_OL_N[o_j][act_fn_j][out_fn_j]+=1;


        return {'transferFn_HL' : transferFn_HL_N, 'transferFn_OL':transferFn_OL_N};

    def getSignatures(self):
        """

        :return: signatures
        """

        weightFns = len(activation_fn.fn_indices);
        nodeFns = len(output_fn.fn_indices);


        #by node position
        transferFn_HL_N = dict();
        transferFn_OL_N = dict();

        #by layer
        transferFns_HL = np.zeros((weightFns,nodeFns),np.float32);
        transferFns_OL = np.zeros((weightFns,nodeFns),np.float32);

        #First order
        weightFns_HL = np.zeros((weightFns),np.float32);
        weightFns_OL = np.zeros((weightFns),np.float32);
        nodeFns_HL = np.zeros((nodeFns),np.float32);
        nodeFns_OL = np.zeros((nodeFns),np.float32);

        #Second order
        Comb = [(w,n) for w in np.arange(1,weightFns+1) for n in np.arange(1,nodeFns+1)];
        numComb = len(Comb);
        #hidden to hidden higher order signatures
        CoexistModelMatHL = np.zeros((numComb,numComb),np.float32);
        CoexistPathMatHL = np.zeros((numComb,numComb),np.float32);
        CoexistLayerMatHL = np.zeros((numComb,numComb),np.float32);

        #hidden to output higher order signatures
        CoexistModelMatOL = np.zeros((numComb,numComb),np.float32);
        CoexistPathMatOL = np.zeros((numComb,numComb),np.float32);
        CoexistLayerMatOL = np.zeros((numComb,numComb),np.float32);

        #connection strength for the hidden to hidden layer
        ConnWeightPathHL = np.zeros((numComb, numComb), np.float32);
        ConnWeightLayerHL = np.zeros((numComb,numComb), np.float32);
        ConnWeightModelHL = np.zeros((numComb,numComb), np.float32);

        #connection strength for the hidden to output layer
        ConnWeightPathOL = np.zeros((numComb, numComb), np.float32);
        ConnWeightLayerOL = np.zeros((numComb,numComb), np.float32);
        ConnWeightModelOL = np.zeros((numComb,numComb), np.float32);

        for i in xrange(len(self.hiddenNodes)):
            transferFn_HL_N[str(i)] = np.zeros((weightFns,nodeFns), np.float32);

        for i in xrange(len(self.outputNodes)):
            transferFn_OL_N[str(i)] = np.zeros((weightFns,nodeFns), np.float32);

        # HIDDEN NODE
        #Caculate the weights
        for h_j,hid_node_j in enumerate(self.hiddenNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            #for each node position
            #transferFn_HL_N[str(h_j)][act_fn_j][out_fn_j]+=1;
            #for each layer
            transferFns_HL[act_fn_j][out_fn_j] +=1;

            #first order
            weightFns_HL[act_fn_j]+=1;
            nodeFns_HL[out_fn_j]+=1;

            #higher order
            #pair the transfer functions
            trans_hj = ((act_fn_j), (out_fn_j));
            for h_i, hid_node_i in enumerate(self.hiddenNodes):

                conn_status = self.connActive_HH[h_i][h_j];
                conn_weight = self.connWeight_IH[h_i][h_j];

                #get hidden node activation and output function
                act_fn_i = activation_fn.getFunctionIndex(hid_node_i.activation_fn);
                out_fn_i = output_fn.getFunctionIndex(hid_node_i.output_fn);

                #pair the transfer functions
                trans_hi = ((act_fn_i), (out_fn_i));

                ## COEXIST IN MODEL
                #if not self connection
                if h_i != h_j :

                    #get transfer functions indices and mark coordinates on coexistence matrix
                    itransFn_i = Comb.index(trans_hi);
                    itransFn_j = Comb.index(trans_hj);

                    ##############################(1) COEXIST IN SAME NETWORK #################
                    CoexistModelMatHL[itransFn_i][itransFn_j] += 1;
                    ConnWeightModelHL[itransFn_i][itransFn_j]+= conn_weight;

                    ## COEXIST IN PATH
                    if conn_status == 1:

                        #get transfer functions indices and mark coordinates on coexistence matrix
                        itransFn_i = Comb.index(trans_hi);
                        itransFn_j = Comb.index(trans_hj);
                        ##############################(2) COEXIST IN SAME PATH #################
                        CoexistPathMatHL[itransFn_i][itransFn_j] += 1;
                        ConnWeightPathHL[itransFn_j][itransFn_j]+= conn_weight;
                    ##############################(3) COEXIST IN SAME LAYER #################
                    CoexistLayerMatHL[itransFn_i][itransFn_j] += 1;
                    ConnWeightLayerHL[itransFn_j][itransFn_j]+=conn_weight;


        # HIDDEN TO OUTPUT CONNECTIONS
        for o_j, out_node_j in enumerate(self.outputNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(out_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(out_node_j.output_fn);

            #transferFn_OL_N[o_j][act_fn_j][out_fn_j]+=1;
            transferFns_OL[act_fn_j][out_fn_j]+=1;
            #first order
            weightFns_OL[act_fn_j]+=1;
            nodeFns_OL[out_fn_j]+=1;

            #second order signatures
            #higher order
            #pair the transfer functions
            trans_oj = ((act_fn_j), (out_fn_j));
            for h_i, hid_node_i in enumerate(self.hiddenNodes):

                conn_status = self.connActive_HO[h_i][o_j];
                conn_weight = self.connWeight_HO[h_i][o_j];

                #get hidden node activation and output function
                act_fn_i = activation_fn.getFunctionIndex(hid_node_i.activation_fn);
                out_fn_i = output_fn.getFunctionIndex(hid_node_i.output_fn);

                #pair the transfer functions
                trans_hi = ((act_fn_i), (out_fn_i));

                ## COEXIST IN MODEL
                #if not self connection


                #get transfer functions indices and mark coordinates on coexistence matrix
                itransFn_i = Comb.index(trans_hi);
                itransFn_j = Comb.index(trans_oj);

                ##############################(1) COEXIST IN SAME NETWORK #################
                CoexistModelMatOL[itransFn_i][itransFn_j] += 1;
                ConnWeightModelOL[itransFn_j][itransFn_j]+= conn_weight;
                ## COEXIST IN PATH
                if conn_status == 1:

                    #get transfer functions indices and mark coordinates on coexistence matrix
                    itransFn_i = Comb.index(trans_hi);
                    itransFn_j = Comb.index(trans_oj);
                    ##############################(2) COEXIST IN SAME PATH #################
                    CoexistPathMatOL[itransFn_i][itransFn_j] += 1;
                    ConnWeightPathOL[itransFn_i][itransFn_j]+=conn_weight;
                ##############################(3) COEXIST IN SAME LAYER #################
                CoexistLayerMatOL[itransFn_i][itransFn_j] += 1;
                ConnWeightLayerOL[itransFn_i][itransFn_j] +=conn_weight



        return {'transferFns_HL_POS' : transferFn_HL_N, 'transferFns_OL_POS':transferFn_OL_N,
                'transferFns_HL_LAY': transferFns_HL, 'transferFns_OL_LAY': transferFns_OL,
                'weightFns_HL': weightFns_HL, 'weightFns_OL': weightFns_OL,
                'nodeFns_HL':nodeFns_HL, 'nodeFns_OL':nodeFns_OL,
                'CoexistPathMatHL':CoexistPathMatHL, 'CoexistPathMatOL': CoexistPathMatOL,
                'CoexistModelMatHL':CoexistModelMatHL,'CoexistModelMatHL':CoexistModelMatOL,
                'CoexistLayerMatHL':CoexistLayerMatHL, 'CoexistLayerMatOL':CoexistLayerMatOL,
                'ConnWeightLayerHL':ConnWeightLayerHL, 'ConnWeightLayerOL':ConnWeightLayerOL,
                'ConnWeightModelHL':ConnWeightModelHL, 'ConnWeightModelOL':ConnWeightModelOL
                    };

    def getLikelihoodMatrices(self):# METHOD SHOULD BE OKAY
        """
        Returns the problem signature for the given model

        @params: None
        @returns: Coexistence likelihood matrix as Numpy Matrix.
        """

        ##### DO NOT FORGET #######################################################################
        #TODO: Don't forget to only copy this method for the Rest of the HypSpaceSpread Experiments
        #TODO: Seperate each propagation method to make things easier to debug e.g. have propagate_IH(),propagate_HH(),and propagate_HO()
        #TODO : update this to repo (3/7/14) - [Not Done ]

        #weight and node functions
        weightFns = len(activation_fn.fn_indices);
        nodeFns = len(output_fn.fn_indices);

        #create transfer function combination
        Comb = [(w,n) for w in np.arange(1,weightFns+1) for n in np.arange(1,nodeFns+1)];
        numComb = len(Comb);
        CoexistModelMatHL = np.zeros((numComb,numComb),np.float32);
        CoexistPathMatHL = np.zeros((numComb,numComb),np.float32);
        CoexistLayerMatHL = np.zeros((numComb,numComb),np.float32);
        CoexistModelMatOL = np.zeros((numComb,numComb),np.float32);
        CoexistPathMatOL = np.zeros((numComb,numComb),np.float32);
        CoexistLayerMatOL = np.zeros((numComb,numComb),np.float32);

        if debug:
            print "* Possible Combinations: ", Comb;
            print "* CoExistenceMat(Model):", CoexistModelMatHL,CoexistLayerMatOL;
            print "* CoExistenceMat(Path):", CoexistPathMatHL,CoexistPathMatOL;
            print "* CoExistenceMat(Layer):", CoexistLayerMatHL, CoexistPathMatOL;
            print "* Weight Functions: ", weightFns;
            print "* Node Functions: ", nodeFns;

        ##### HIDDEN TO HIDDEN UNITS #####

        #Caculate the weights
        for h_j,hid_node_j in enumerate(self.hiddenNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            #pair the transfer functions
            trans_hj = ((act_fn_j), (out_fn_j));


            for h_i, hid_node_i in enumerate(self.hiddenNodes):

                #get the hidden node
                conn_status = self.connActive_HH[h_i][h_j];

                #get hidden node activation and output function
                act_fn_i = activation_fn.getFunctionIndex(hid_node_i.activation_fn);
                out_fn_i = output_fn.getFunctionIndex(hid_node_i.output_fn);

                #pair the transfer functions
                trans_hi = ((act_fn_i), (out_fn_i));

                ## COEXIST IN MODEL
                #if not self connection
                if h_i != h_j :

                    #get transfer functions indices and mark coordinates on coexistence matrix
                    itransFn_i = Comb.index(trans_hi);
                    itransFn_j = Comb.index(trans_hj);

                    ##############################(1) COEXIST IN SAME NETWORK #################
                    CoexistModelMatHL[itransFn_i][itransFn_j] += 1;

                    ## COEXIST IN PATH
                    if conn_status == 1:

                        #get transfer functions indices and mark coordinates on coexistence matrix
                        itransFn_i = Comb.index(trans_hi);
                        itransFn_j = Comb.index(trans_hj);
                        ##############################(2) COEXIST IN SAME PATH #################
                        CoexistPathMatHL[itransFn_i][itransFn_j] += 1;

                    ##############################(3) COEXIST IN SAME LAYER #################
                    CoexistLayerMatHL[itransFn_i][itransFn_j] += 1;


        ##### HIDDEN TO OUTPUT UNITS #####
        #Caculate the weights
        for o_j,out_node_j in enumerate(self.outputNodes):

            #get hidden node activation and output function
            act_fn_jhh = activation_fn.getFunctionIndex(out_node_j.activation_fn);
            out_fn_jhh = output_fn.getFunctionIndex(out_node_j.output_fn);

            #pair the transfer functions
            trans_hj = ((act_fn_jhh), (out_fn_jhh));

            #Incoming connections from hidden to output nodes
            for h_i, hid_node_i in enumerate(self.hiddenNodes):

                #get the hidden node
                conn_status = self.connActive_HO[h_i][o_j];

                #get hidden node activation and output function
                act_fn_i = activation_fn.getFunctionIndex(hid_node_i.activation_fn);
                out_fn_i = output_fn.getFunctionIndex(hid_node_i.output_fn);

                #pair the transfer functions
                trans_hi = ((act_fn_i), (out_fn_i));

                ## COEXIST IN MODEL
                #get transfer functions indices and mark coordinates on coexistence matrix
                itransFn_i = Comb.index(trans_hi);
                itransFn_j = Comb.index(trans_hj);

                ##############################(1) COEXIST IN SAME NETWORK #################
                CoexistModelMatOL[itransFn_i][itransFn_j] += 1;

                ## COEXIST IN PATH
                if conn_status == 1:
                    #pair the transfer functions
                    trans_hi = ((act_fn_i), (out_fn_i));

                    #get transfer functions indices and mark coordinates on coexistence matrix
                    itransFn_i = Comb.index(trans_hi);
                    itransFn_j = Comb.index(trans_hj);
                    ##############################(2) COEXIST IN SAME PATH #################
                    CoexistPathMatOL[itransFn_i][itransFn_j] += 1;

                    ##############################(3) COEXIST IN SAME LAYER #################
                    CoexistLayerMatOL[itransFn_i][itransFn_j] += 1;


        return {'onPathHL': CoexistPathMatHL,'inLayerHL': CoexistLayerMatHL, 'inModelHL': CoexistModelMatHL,
                'onPathOL': CoexistPathMatOL,'inLayerOL': CoexistLayerMatOL, 'inModelOL': CoexistModelMatOL };



    def getNodesNEdges(self):
        """  Returns the nodes and edges """


        #return object
        NodesNEdges = dict()
        NodesNEdges['nodes'] = [];
        NodesNEdges['edges'] = [];


        # HIDDEN TO HIDDEN CONNECTIONS
        for h_j,hid_node_j in enumerate(self.hiddenNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(hid_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(hid_node_j.output_fn);

            #pair the transfer functions
            trans_hj = (act_fn_j, out_fn_j);


            for h_i, hid_node_i in enumerate(self.hiddenNodes):

                #get the hidden node
                conn_status = self.connActive_HH[h_i][h_j];

                #get hidden node activation and output function
                act_fn_i = activation_fn.getFunctionIndex(hid_node_i.activation_fn);
                out_fn_i = output_fn.getFunctionIndex(hid_node_i.output_fn);

                #pair the transfer functions
                trans_hi = (act_fn_i, out_fn_i);

                ## COEXIST IN MODEL
                #if not self connection
                if h_i != h_j and conn_status :
                    NodesNEdges['nodes'].append(trans_hi);
                    NodesNEdges['nodes'].append(trans_hj);
                    NodesNEdges['edges'].append((trans_hi,trans_hj));


        ##### HIDDEN TO OUTPUT UNITS #####
        #Caculate the weights
        for o_j,out_node_j in enumerate(self.outputNodes):

            #get hidden node activation and output function
            act_fn_j = activation_fn.getFunctionIndex(out_node_j.activation_fn);
            out_fn_j = output_fn.getFunctionIndex(out_node_j.output_fn);

            #pair the transfer functions
            trans_hj = (act_fn_j, out_fn_j);

            #Incoming connections from hidden to output nodes
            for h_i, hid_node_i in enumerate(self.hiddenNodes):

                #get the hidden node
                conn_status = self.connActive_HO[h_i][o_j];

                #get hidden node activation and output function
                act_fn_i = activation_fn.getFunctionIndex(hid_node_i.activation_fn);
                out_fn_i = output_fn.getFunctionIndex(hid_node_i.output_fn);

                #pair the transfer functions
                trans_hi = (act_fn_i, out_fn_i);

                ## COEXIST IN PATH
                if conn_status == 1:
                    NodesNEdges['nodes'].append(trans_hi);
                    NodesNEdges['nodes'].append(trans_hj);
                    NodesNEdges['edges'].append([trans_hi,trans_hj]);


        #debug
        if debug or True:

            print NodesNEdges;



        return NodesNEdges;