import os, sys, inspect, traceback;
import exceptions as e;
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

sys.path.insert(0, "../");  #use default settings
sys.path.insert(0, "../core");
sys.path.insert(0, "../visualisation");
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../cuda");
sys.path.insert(0, "../tools");

"""
pyNDMOptim

Optimisation for pyNDMs using the DEAP library.
This design is based on the hierarchical nature of the neural network search space.

@Author: Abdullahi Adam
@mail:(abdullah.adam89@hotmail.com)
@Pre-requisite Modules: pytools, pycuda, numpy, pyDEAP and matplotlib
"""


#Outputs
debug = False;
verbose = False;

#####imports######
import random;
import numpy as np;
from deap import tools, base, creator;
import ndmModel;
import datasets;

#Transfer functions
import activation_fn as inFn; #input combination functions
import output_fn as outFn; #output function

# #genome and network
# from optimEnDec import *;
# from ndm import ndm;
# import netParams;
import constants;

##TODO : Write a node class (Completed)
##TODO : Write a model class
##TODO : Write a bias class



#TODO: consider adding some model preference rules.
#TODO: Generating the nodes in the node population, how do we get a good spread of nodes?

####### HELPER FUNCTIONS #########################
def initFullConnectivity(connActive, leftlayersize, rightlayersize):
    """
    Generate a full connectivity between two layers
    """

    M = np.ones((leftlayersize, rightlayersize));

    return connActive(M);


def initRandomConnectivity(connActive, leftlayersize, rightlayersize):
    """
    Generates a random connectivity matrix between two layers
    """
    M = np.zeros((leftlayersize, rightlayersize));

    for i in xrange(len(M)):
        for j in xrange(len(M[i])):
            M[i][j] = random.randint(0, 1);

    return connActive(M);


def initRandomWeights(connWeights, leftlayersize, rightlayersize):
    """
    Generates a random matrix of connection weights two layers
    """

    M = np.random.rand(leftlayersize, rightlayersize);

    return connWeights(M);


def initZeroConnectivity(connActive, leftlayersize, rightlayersize):
    """
    Generates a connectivity matrix with no connections
    """

    M = np.zeros(leftlayersize,rightlayersize);

    return connActive(M);


NODEPARAMS = 5;
# 1 -FN param1
# 2 -FN Param2
# 3 -FN param3
# 4 -Bias
# 5 -Bias Weight
MODELPARAMS = 3;
# 1 -  USE Lateral connections : BOOL
# 2 - USE Recurrent (Elamn): BOOL
# 3 - Use Gauss noise on inputs : BOOL


class pyNDMOptim:


    def __init__(self, Bias = None):
        """
            Initialise
        """


        self.current_iter = 0;

        self.train_set = datasets.XOR;
        self.validation_set = datasets.XOR;
        self.test_set = datasets.XOR;


        self.tbox = base.Toolbox();
        self.pop = {
            'model_pop': [],
            'node_pop': [],
            'connActive_IH_pop': [],
            'connActive_HO_pop': [],
            'connActive_HH_pop': [],
            'connWeights_IH_pop': [],
            'connWeights_HO_pop': [],
            'connWeights_HH_pop': [],
        };

        self.fitness_landscape ={
            #NODE
            'node-model': None,
            # 'node-connActiveIH': None,
            # 'node-connActiveHH': None,
            # 'node-connActiveHO': None,
            # 'node-weightIH': None,
            # 'node-weightHH': None,
            # 'node-weightHO': None,
            #MODEL
            'connActiveIH-model': None,
            'connActiveHH-model': None,
            'connActiveHO-model': None,
            'weightIH-model': None,
            'weightHH-model': None,
            'weightHO-model': None,

            #WEIGHT and CONNECTIVITY
            # 'weightIH-connActiveIH': None,
            # 'weightHH-connActiveHH': None,
            # 'weightHO-connActiveHO': None,





        };

        if Bias != None:
            pass;
        else:
            #Optimisation params - BIAS
            self.params = dict();
            self.params['targer_err'] = 0.0;
            self.params['NGEN'] = 100;
            self.params['numInputNodes'] = 2
            self.params['numHiddenNodes'] = 2;
            self.params['numOutputNodes'] = 1;
            self.params['select_mut'] = 1;
            self.params['select_nextGen'] = 1;
            self.params['select_crossOver'] = 1;
            self.params['select_diffEvo'] = 1;
            self.params['initConnectivity'] = 'full'; #connectivity {'random', 'full'}
            self.params['initTransferFn'] = 'random'; #random: randomly creates transfer functions, and doesn't account for duplication, #all: creates all possible combinations of transfer functions
            self.params['output_fns'] = [outFn.identity_fn, outFn.sigmoid_fn, outFn.gaussian_fn]; #output functions
            self.params['activation_fns'] = [inFn.inner_prod, inFn.euclidean_dist, inFn.manhattan_dist]; #weight/input functions
            self.params['node_pop_size'] = 10;
            self.params['conn_pop_size'] = 10;
            self.params['model_pop_size'] = 10; #ideally 10
            self.params['prodConstantMax'] = 5;
            self.params['greedThreshold'] = 3;
            self.params['percentNextGen'] = 0.5; #percentage of the population to select for next generation
            # self.params['iterMax'] = optimParams.max_iter ; #maximum iterations
            # self.params['DEIterMax'] = optimParams.deIter; #max differential evolution iterations
            # self.params['target_error'] = optimParams.target_err; # targer error
            # self.params['pop_size'] = optimParams.pop_size; #size of population
            # self.params['next_genSize'] = optimParams.next_genSize; #size of next generation
            self.params['prob_crossOver'] = True; #probability of crossover
            self.params['prob_co_indp'] = 0.6; #probability of crossing over individual parameter
            self.params['prob_mutation'] = 0.7; #probablity of mutation
            self.params['prob_mut_indp'] = 0.2; #probability of mutating individual parameters
            self.params['alpha'] = 0.2; #significant of velocities for differential evolution
            self.params['param_min'] = -1.0; #parameter min
            self.params['param_max'] = 1.0; #parameter max
            self.params['do_diffEvolution'] = True; #does differential evolution
            self.params['do_crossOver'] = True; #cross over
            self.params['do_mutation'] = True; #mutation
            self.params['gaus_mut_mean'] = 0.0; #mutaion mean
            self.params['gaus_mut_std'] = 0.2; #mutation std. deviation
            # self.params['mutation_range'] = optimParams.mutation_range; #mutation range
            # self.params['percent_trim'] = optimParams.percent_trim; #percentage of solutions to eliminate


    def createPopulations(self):
        """ Creates the populations of the neural network subcomponents for coevolution"""

        #Fitness class
        creator.create("fitness", base.Fitness, weights=(-1, -1));

        #Node class
        creator.create("node", np.ndarray,
                       fitness=creator.fitness,
                       id=None ,
                       node_type=None,
                       activation_fn=None,  #or weight fn
                       output_fn=None  #or output fn
        );

        #Connection class
        creator.create("ConnActive", np.ndarray,
                       fitness=creator.fitness,
                       from_layer_id = None,
                       to_layer_id = None,
                       id =  None);

        creator.create("ConnWeights", np.ndarray,
                       fitness=creator.fitness,
                       from_layer_id = None,
                       to_layer_id = None,
                       id = None);

        #Model Class
        creator.create("model", np.ndarray,
                       fitness = creator.fitness,
                        prodConstant = None,
                        id = None);

        #Register to the toolbox
        # - Attributes
        self.tbox.register("float_attr", random.random);
        self.tbox.register("bool_attr", random.randint, 0, 1);

        ################################ 1- Node ################################################################
        self.tbox.register("node", tools.initRepeat, creator.node, self.tbox.float_attr, NODEPARAMS);

        ################################ Connections/Topology #################################################
        # - 2- Connectivity (Input to hidden units)
        if self.params['initConnectivity'] == 'random':
            self.tbox.register("connActive_IH", initRandomConnectivity, creator.ConnActive,
                               self.params['numInputNodes'],
                               self.params['numHiddenNodes']);
            self.tbox.register("connActive_HO", initRandomConnectivity, creator.ConnActive,
                               self.params['numHiddenNodes'],
                               self.params['numOutputNodes']);

        elif self.params['initConnectivity'] == 'full':
            self.tbox.register("connActive_IH", initFullConnectivity, creator.ConnActive,
                               self.params['numInputNodes'],
                               self.params['numHiddenNodes']);
            self.tbox.register("connActive_HO", initFullConnectivity, creator.ConnActive,
                                self.params['numHiddenNodes'],
                               self.params['numOutputNodes']);

        # - 3- Connection Weights (Input to hidden layer, hidden layer to output layer)
        self.tbox.register("connWeights_IH", initRandomWeights, creator.ConnWeights,
                           self.params['numInputNodes'],
                           self.params['numHiddenNodes']);
        self.tbox.register("connWeights_HO", initRandomWeights, creator.ConnWeights,
                           self.params['numHiddenNodes'],
                           self.params['numOutputNodes']);

        # - Connectivity (Hidden to hidden layer)
        self.tbox.register("connActive_HH", initZeroConnectivity, creator.ConnActive,
                           self.params['numHiddenNodes'],
                           self.params['numHiddenNodes']);

        # - Connection weights (hidden to hidden layer)
        self.tbox.register("connWeights_HH", initRandomWeights, creator.ConnWeights,
                           self.params['numHiddenNodes'],
                           self.params['numHiddenNodes']);

        ####################################### Model ############################################################
        self.tbox.register("model", tools.initRepeat, creator.model, self.tbox.bool_attr,MODELPARAMS)

        ################################# Create populations#######################################################

        # - Nodes
        if self.params['initTransferFn'] == 'all': #i.e. all possible combinations (no duplicates)
            node_pop_size = len(self.params['activation_fns']) * len(self.params['output_fns']);
        elif self.params['initTransferFn'] == 'random': #i.e. random, with possibility of duplicates
            node_pop_size = self.params['node_pop_size'];
        #- node pop####
        self.tbox.register("node_pop",tools.initRepeat, list, self.tbox.node, node_pop_size);
        # - Connections pop###
        self.tbox.register("connActive_IH_pop", tools.initRepeat, list, self.tbox.connActive_IH, self.params['conn_pop_size']);
        self.tbox.register("connActive_HO_pop", tools.initRepeat, list, self.tbox.connActive_HO, self.params['conn_pop_size']);
        self.tbox.register("connActive_HH_pop", tools.initRepeat, list, self.tbox.connActive_HH, self.params['conn_pop_size']);
        self.tbox.register("connWeights_IH_pop", tools.initRepeat, list, self.tbox.connWeights_IH, self.params['conn_pop_size']);
        self.tbox.register("connWeights_HO_pop", tools.initRepeat, list, self.tbox.connWeights_HO, self.params['conn_pop_size']);
        self.tbox.register("connWeights_HH_pop", tools.initRepeat, list, self.tbox.connWeights_HH, self.params['conn_pop_size']);

        ### Model pop###
        self.tbox.register("model_pop", tools.initRepeat, list, self.tbox.model, self.params['model_pop_size']);

        ###Create pop###
        self.pop['node_pop'] = self.tbox.node_pop();
        self.pop['connWeights_IH_pop'] = self.tbox.connWeights_IH_pop();
        self.pop['connWeights_HH_pop'] = self.tbox.connWeights_HH_pop();
        self.pop['connWeights_HO_pop'] = self.tbox.connWeights_HO_pop();
        self.pop['connActive_IH_pop'] = self.tbox.connActive_IH_pop();
        self.pop['connActive_HH_pop'] = self.tbox.connActive_HH_pop();
        self.pop['connActive_HO_pop'] = self.tbox.connActive_HO_pop();
        self.pop['model_pop'] = self.tbox.model_pop();

        #Initialise nodes with transfer functions
        #TODO: might want to include other methods to intialiasing later on for experiments
        self.init_tf_fn_random(); #TODO: EXPERIMENNT - Trasfer function optimisation {random or all}
        self.init_connections();
        self.init_models();
        self.init_evolutionary_operators();


    def init_evolutionary_operators(self):
        """
        initialise the operators
        """
        #MUTATE WEIGHTS
        self.tbox.register("mutate_weights", tools.mutGaussian,
                           mu = self.params['gaus_mut_mean'],
                           sigma = self.params['gaus_mut_std'],
                           indpb = self.params['prob_mut_indp']);


        #NODE FNPARAMS
        self.tbox.register("mutate_fnParams", tools.mutGaussian,
                           mu = self.params['gaus_mut_mean'],
                           sigma = self.params['gaus_mut_std'],
                           indpb = self.params['prob_mut_indp']);
        #CONNECTIVITY
        self.tbox.register("mutate_conn", tools.mutFlipBit,
                           indpb=self.params['prob_mut_indp']);

        #MODEL
        self.tbox.register("mutate_model", tools.mutFlipBit,
                           mu = self.params['gaus_mut_mean'],
                           sigma = self.params['gaus_mut_std'],
                           indpb = self.params['prob_mut_indp']);



    def init_tf_fn_random(self,nodes = None):
        """
        Initialise the nodes with the set of transfer functions available
        """
        i = 0;
        for node in self.pop['node_pop']:

            #output function
            outfn = random.randint(0,len(self.params['output_fns'])-1);
            infn = random.randint(0,len(self.params['activation_fns'])-1);

            ## Reference the function directly
            node.output_fn = self.params['output_fns'][outfn];
            node.activation_fn = self.params['activation_fns'][infn];
            node.id = i;
            i += 1;

            if debug:
                print ">>", node.output_fn, node.activation_fn;


    def init_transferFns(self,nodes = None):
        """
        Initialise the transfer functions
        """

        if self.pop['node_pop'] < len(self.params['activation_fns']) * len(self.params['output_fns']):
            raise Exception("Not enough nodes for init_transferFns");

        tfs_comb = [(w,n) for w in self.params['activation_fns'] for n in self.params['output_fns']];

        i = 0;
        for node,tf in zip(self.pop['node_pop'],tfs_comb):
                #output functions
                node.activation_fn = tf[0];
                node.output_fn = tf[1];
                node.id = id;
                i += 1;

        if debug:
            for node in self.pop['node_pop']:
                print ">>output_fns:", node.output_fn;
                print ">> activation_fn:",node.activation_fn;

    def init_connections(self):
        """
        Initialise the connections
        """

        #Input to hidden
        i = 0;
        for connActive,connWeight in zip(self.pop['connActive_IH_pop'], self.pop['connWeights_IH_pop']):

            connActive.id = i;
            connWeight.id = i;

            i = i + 1;

        #Hidden to Hidden (Lateral)
        i = 0;
        for connActive,connWeight in zip(self.pop['connActive_HH_pop'], self.pop['connWeights_HH_pop']):

            connActive.id = i;
            connWeight.id = i;

            i = i + 1;

        #Hidden to output
        i = 0;
        for connActive,connWeight in zip(self.pop['connActive_HO_pop'], self.pop['connWeights_HO_pop']):

            connActive.id = i;
            connWeight.id = i;

            i = i + 1;


    def init_models(self):
        """
        Gives the models ids
        """

        i = 0;
        for m in self.pop['model_pop']:
            m.id = i;
            m.prodConstant = random.randint(1,self.params['prodConstantMax']);

            #increament
            i = i + 1;


    def evaluate(self, components):
        """
        Evaluates the specie's
        """

        model = ndmModel.ndmModel(self.params['numInputNodes'],
                                  self.params['numHiddenNodes'],
                                  self.params['numOutputNodes'],
                                  components);

        f = model.evaluate(self.train_set);

        return f;


    def diff_evolve(self):
        """
        Performs differential evolution on some
        """
        #TODO: complete this function to perform differential evolution on each component
        #TODO: CONSIDER IMPLEMENTING GPU CODE TO HANDLE THIS


    def crossOver(self):
        """
        Cross over components of the population
        """
        #TODO: complete this function and test


    def mutate(self, pop = None):
        """
        Mutates the components of the neural network
        """
        #TODO: complete this function (DONE , but TEST)
        #TODO: consider having seperate mutation parameters for the components [EXPERIMENT]
        #TODO: consider making the selection process less greedy [EXPERIMENT - Funneling, convergence]

        offspring_connIH = self.mutate_connIH();
        offspring_connHH = self.mutate_connHH();
        offspring_connHO = self.mutate_connHO();
        offspring_node = self.mutate_nodes();
        offspring_model = self.mutate_models();
        offspring_weightsIH = self.mutate_weightsIH();
        offspring_weightsHH = self.mutate_weightsHH();
        offspring_weightsHO = self.mutate_weightsHO();

        return {'model': offspring_model,
                'node': offspring_node,
                'connIH': offspring_connIH,
                'connHH': offspring_connHH,
                'connHO': offspring_connHO,
                'weightsIH': offspring_weightsIH,
                'weightsHH': offspring_weightsHH,
                'weightsHO': offspring_weightsHO
        };


    def co_weightsIH(self, pop = None):
        """
        Cross over the weights between the input and hidden layer
        """

        if pop == None:
            pop = self.pop['connWeights_IH_pop'];

        parents = tools.selTournament(pop,2,)






    def mutate_weightsIH(self, pop = None):
        """
        Mutates the weight components (Input to Hidden layer weights)
        """

        if verbose:
            print ">> Mutation -> WeightIH";

        if pop == None:
            pop = self.pop['connWeights_IH_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            self.tbox.mutate_weights(mutant);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;


    def mutate_weightsHH(self, pop= None):
        """
        Mutates the weight components (Hidden to Hidden layer weights)
        """

        if verbose:
            print ">> Mutation -> weightHH";

        if pop == None:
            pop = self.pop['connWeights_HH_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            self.tbox.mutate_weights(mutant);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;


    def mutate_weightsHO(self, pop = None):
        """
        Mutates the weight components (Hidden to Output layers weights)
        """

        if verbose:
            print ">> Mutation -> WeightHO";

        if pop == None:
            pop = self.pop['connWeights_HO_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            self.tbox.mutate_weights(mutant);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;


    def mutate_connIH(self, pop = None):
        """
        Mutates the connection components (Input to Hidden layer weights)
        """

        if verbose:
            print ">> Mutation -> connIH";

        if pop == None:
            pop = self.pop['connActive_IH_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            for i in xrange(len(mutant)):
                self.tbox.mutate_conn(mutant[i]);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;

    def mutate_connHH(self, pop= None):
        """
        Mutates the connection components (Hidden to Hidden layer weights)
        """

        if verbose:
            print ">> Mutation -> connHH";

        if pop == None:
            pop = self.pop['connActive_HH_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            for i in xrange(len(mutant)):
                self.tbox.mutate_conn(mutant[i]);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;


    def mutate_connHO(self, pop = None):
        """
        Mutates the connection components (Hidden to Output layers weights)
        """

        if verbose:
            print ">> Mutation -> connHO";

        if pop == None:
            pop = self.pop['connActive_HO_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            for i in xrange(len(mutant)):
                self.tbox.mutate_conn(mutant[i]);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;

    def mutate_nodes(self, pop = None):
        """
        Mutates the nodes - fnParameters, including bias and bias weight
        """
        #TODO: consider mutating the prodCosntant [EXPERIMENT]
        if verbose:
            print ">> Mutation -> Nodes";

        if pop == None:
            pop = self.pop['node_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            self.tbox.mutate_fnParams(mutant);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;


            return mutant;

        else:
            return None;

    def mutate_models(self, pop = None):
        """
        Mutates the models  - {lateral connection, gaussian mutation of inputs, context layer}

        """
        if verbose:
            print ">> Mutation -> Models";

        if pop == None:
            pop = self.pop['model_pop'];

        if random.random() < self.params['prob_mutation']:
            ind = tools.selRandom(pop,1).pop();
            mutant = self.tbox.clone(ind);
            self.tbox.mutate_conn(mutant);
            mutant.id = None;
            del mutant.fitness.values;
            if verbose:
                print "parent", ind;
                print "child", mutant;

            return mutant;

        else:
            return None;




    def select_nextGen(self):
        """
        Selects the components for the next generation
        """

        #TODO: Complete this function
        next_gen_pop = dict();


        model_pick = int(self.params['percentNextGen'] * self.params['model_pop_size']);
        conn_pick = int(self.params['percentNextGen'] * self.params['conn_pop_size']);


        ##Select node
        next_gen_pop['node_pop'] = self.pop['node_pop']; #move all to next generation

        ##Select weights
        next_gen_pop['connWeights_IH_pop'] = tools.selBest(self.pop['connWeights_IH_pop'],conn_pick);
        next_gen_pop['connWeights_HH_pop'] = tools.selBest(self.pop['connWeights_HH_pop'],conn_pick);
        next_gen_pop['connWeights_HO_pop'] = tools.selBest(self.pop['connWeights_HO_pop'],conn_pick);

        ##Select connectivity
        next_gen_pop['connActive_IH_pop'] = tools.selBest(self.pop['connActive_IH_pop'],conn_pick);
        next_gen_pop['connActive_HH_pop'] = tools.selBest(self.pop['connActive_HH_pop'],conn_pick);
        next_gen_pop['connActive_HO_pop'] = tools.selBest(self.pop['connActive_HO_pop'],conn_pick);

        ##select model
        next_gen_pop['model_pop'] = tools.selBest(self.pop['model_pop'],model_pick);

        return next_gen_pop;


    def optim(self):
        """

        """

        ##Create population of species to work with
        self.createPopulations();

        #select representatives
        repr = dict();
        repr['hidden_nodes'] = tools.selRandom(self.pop['node_pop'],
                                                                 self.params['numHiddenNodes']);
        repr['out_nodes'] = tools.selRandom(self.pop['node_pop'],
                                                  self.params['numOutputNodes']);
        #select random connections and weights
        repr['model'] = tools.selRandom(self.pop['model_pop'], 1);
        repr['connActive_IH'] = tools.selRandom(self.pop['connActive_IH_pop'], 1);
        repr['connActive_HH'] = tools.selRandom(self.pop['connActive_HH_pop'], 1);
        repr['connActive_HO'] = tools.selRandom(self.pop['connActive_HO_pop'], 1);
        repr['connWeights_IH'] = tools.selRandom(self.pop['connWeights_IH_pop'], 1);
        repr['connWeights_HH'] = tools.selRandom(self.pop['connWeights_HH_pop'], 1);
        repr['connWeights_HO'] = tools.selRandom(self.pop['connWeights_HO_pop'], 1);
        # print "Representatives", repr;

        #Co-op Coevolution
        g = 0;
        for g in  xrange(self.params['NGEN']):

            #Go through components population
            for comp_key  in self.pop.keys():

                next_repr = self.tbox.clone(repr);

                if comp_key == 'node_pop':
                    #Output units
                    for ind in self.pop['node_pop']:
                        #clone representative
                        components = self.tbox.clone(repr);
                        #evaluate the components population
                        components['out_nodes'] = [ind];
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "n", ind.fitness.values[0];
                        del components;

                    self.pop['node_pop'] = tools.selTournament(self.pop['node_pop'], len(self.pop['node_pop']), 3);
                    next_repr['out_nodes'] = tools.selBest(self.pop['node_pop'],1);

                    # #clone
                    # components = self.tbox.clone(repr);
                    # #Hidden units
                    # for nodes in self.pop['node_pop']:
                    #     #get representatives
                    #     node_repr = self.tbox.clone(repr['hidden_nodes']);
                    #     for i, n in enumerate(nodes):
                    #         r1 = (node_repr[:i]);
                    #         r2 = (node_repr[i+1:]);
                    #         r1.extend(r2);
                    #         print "Representative:", r1;
                    #         #evaluate the components population
                    #         components['hidden_nodes'] = r1.extend([n]);
                    #         #assign fitness
                    #         ind.fitness.values = self.evaluate(components),-1;
                    #         # print "fitness:", ind.fitness.values[0];


                if comp_key == 'connActive_IH_pop':

                    #CONN IH
                    for ind in self.pop['connActive_IH_pop']:
                        #clone representative
                        components = self.tbox.clone(repr);
                        #evaluate the components population
                        components['connActive_IH'] = ind;
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "fitness:", ind.fitness.values[0];
                        del components;
                self.pop['connActive_IH_pop'] = tools.selTournament(self.pop['connActive_IH_pop'],
                                                                    len(self.pop['connActive_IH_pop']), 3);
                next_repr['connActive_IH'] = tools.selBest(self.pop['connActive_IH_pop'],1);

                if comp_key == 'connActive_HH_pop':

                    #CONN IH
                    for ind in self.pop['connWeights_HH_pop']:
                        #clone representative
                        components = self.tbox.clone(repr);
                        #evaluate the components population
                        components['connActive_HH'] = ind;
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "fitness:", ind.fitness.values[0];
                        del components;

                self.pop['connActive_HH_pop'] = tools.selTournament(self.pop['connActive_HH_pop'],
                                                                    len(self.pop['connActive_HH_pop']), 3);
                next_repr['connActive_HH'] = tools.selBest(self.pop['connActive_HH_pop'],1);


                if comp_key == 'connActive_HO_pop':

                    #CONN IH
                    for ind in self.pop['connWeights_HO_pop']:
                        #clone representative
                        components = self.tbox.clone(repr);
                        #evaluate the components population
                        components['connActive_HO'] = ind;
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "fitness:", ind.fitness.values[0];

                        del components;

                self.pop['connActive_HO_pop'] = tools.selTournament(self.pop['connActive_HO_pop'],
                                                                    len(self.pop['connActive_HO_pop']), 3);
                next_repr['connActive_HO'] = tools.selBest(self.pop['connActive_HO_pop'],1);


                if comp_key == 'connWeights_IH_pop':

                    #CONN IH
                    for ind in self.pop['connWeights_IH_pop']:
                        #clone representative
                        components = self.tbox.clone(repr);
                        #evaluate the components population
                        components['connWeights_IH'] = ind;
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "fitness:", ind.fitness.values[0];

                        del components;

                self.pop['connWeights_IH_pop'] = tools.selTournament(self.pop['connWeights_IH_pop'],
                                                                    len(self.pop['connWeights_IH_pop']), 3);
                next_repr['connWeights_IH'] = tools.selBest(self.pop['connWeights_IH_pop'],1);


                if comp_key == 'connWeights_HH_pop':
                    #clone representative
                    components = self.tbox.clone(repr);
                    #CONN IH
                    for ind in self.pop['connWeights_HH_pop']:
                        #evaluate the components population
                        components['connWeights_HH'] = ind;
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "fitness:", ind.fitness.values[0];

                self.pop['connWeights_HH_pop'] = tools.selTournament(self.pop['connWeights_HH_pop'],
                                                                    len(self.pop['connWeights_HH_pop']), 3);
                next_repr['connWeights_HH'] = tools.selBest(self.pop['connWeights_HH_pop'],1);

                if comp_key == 'connWeights_HO_pop':
                    #clone representative
                    components = self.tbox.clone(repr);
                    #CONN IH
                    for ind in self.pop['connWeights_HO_pop']:
                        #evaluate the components population
                        components['connWeights_HO'] = ind;
                        #assign fitness
                        ind.fitness.values = self.evaluate(components),-1;
                        print "fitness:", ind.fitness.values[0];

                self.pop['connWeights_HO_pop'] = tools.selTournament(self.pop['connWeights_HO_pop'],
                                                                    len(self.pop['connWeights_HO_pop']), 3);
                next_repr['connWeights_HO'] = tools.selBest(self.pop['connWeights_HO_pop'],1);


            repr = next_repr;

##### TEST ########
# p = pyNDMOptim();
# p.optim();
# sys.exit();

#TODO: Coevolution of neural computation paths
#TODO: