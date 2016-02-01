__author__ = 'Abdullah'

import sys;

sys.path.insert(0, "../");  #use default settings
sys.path.insert(0, "../core");
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../tools");
sys.path.insert(0, "../cuda");
sys.path.insert(0, "../visualisation");




debug= False;
from pandas import DataFrame;
import ndmCoevoBase;
import coevoOptim;
import commons;
from ndmModel import ndmModel;

import activation_fn as inFn;
import output_fn as outFn;
import datasets;
import commons;
import time;
import cPickle as pickle;
#import visualisation.visualiseOutputs2D as vis2d;
from deap import tools, algorithms;
import matplotlib.pyplot as plt;


#import visualiseNet
try:
    from visualiseNDMNet import *;
    visualiseNet = False;
except ImportError:
    visualiseNet = True;


####
## CHANGES
# #TODO: Added connHO evaluation. (removed)
# #TODO: Added

#TODO : PENALISE MODELS WITH NO DISCRIMINATORY ABILITY (DONE)
#TODO:
class ndmCoevoOptim:

    def __init__(self, Bias = None, noTraining = False, dataset_name = '',
                 train_set = None, valid_set = None, test_set = None,netConfig = None,
                 hl_output_fns = None, hl_activation_fns = None,
                 ol_activation_fns = None, ol_output_fns = None, initConnectivity = None,
                 noIntraLayerConnHH = None,noContextLayer = None, NGEN = None):
        """
            Initialise
        """

        #statisitics
        #-fitness
        stats_fitness = tools.Statistics(key = lambda sol: sol.fitness.values);
        #-connectivity statistics
        self.conn_stats = tools.MultiStatistics(fitness = stats_fitness);
        #-weights statistics
        self.weights_stats = tools.MultiStatistics(fitness = stats_fitness);
        #-hidden nodes statistics
        self.hl_stats = tools.MultiStatistics(fitness = stats_fitness);
        #-output node statistics
        self.ol_stats = tools.MultiStatistics(fitness = stats_fitness);
        #-model
        stats_lateral_conn = tools.Statistics(key = lambda model: model[0]); # Lateral connections
        stats_context_layer = tools.Statistics(key = lambda model: model[1]); # Context layer
        stats_gauss_noise = tools.Statistics(key = lambda model: model[2]); # Gaussian Noise

        self.model_stats = tools.MultiStatistics(lateral_conn = stats_lateral_conn,
                                                context_layer = stats_context_layer,
                                                gauss_noise = stats_gauss_noise,
                                                fitness = stats_fitness);

        self.log = tools.Logbook();

        #params
        self.snapshot_interval = 1;
        self.snapshot = {
          'hidden_layer': [float('inf')],
          'output_layer': [float('inf')],
          'topology': [float('inf')],
          'weights': [float('inf')],
          'model': [float('inf')]
        };
        self.current_iter = 0;
        self.best_cost = float('inf');
        self.best_cost_val = float('inf');
        self.best_cost_test = float('inf');

        self.best_cost_lst = [];
        self.best_cost_val_lst = [];

        self.best_sol = None;
        self.best_sol_val = None;
        self.best_sol_test = None;
        self.best_model = None;
        self.plot_interval = 2;


        self.comp_best_cost = {
          'model': [float('inf')],
          'hidden_nodes': dict(),
          'out_nodes': dict(),
          'connActive_IH': [float('inf')],
          'connActive_HH': [float('inf')],
          'connActive_HO': [float('inf')],
          'connWeights_IH': [float('inf')],
          'connWeights_HH': [float('inf')],
          'connWeights_HO': [float('inf')],

        };

        #DATASET
        # proben1 = benchmark.proben1_bechmark();

        self.dataset_name = dataset_name;


        if train_set != None:
            self.train_set = train_set;
        else:
            raise Exception("No Training set given")
            print ">> ";
           # self.train_set = datasets.XOR#proben1.heart()['train'];

        if test_set != None:
            self.test_set = test_set;
        else:
            raise Exception("No Test set given")
            print ">> Using XOR dataset - test ";
           # self.test_set = datasets.XOR#proben1.heart()['test'];

        if valid_set != None:
            self.validation_set = valid_set;
        else:
            print ">> No validation set given";
           # self.validation_set = datasets.XOR;

        #POPULATION
        self.pop = {
            'model_pop': [],
            'hidden_nodes': dict(),
            'out_nodes': dict(),
            'connActive_IH_pop': [],
            'connActive_HO_pop': [],
            'connActive_HH_pop': [],
            'connWeights_IH_pop': [],
            'connWeights_HO_pop': [],
            'connWeights_HH_pop': [],
        };

        #REPRESENTATIVES
        self.represntatives = None;

        #set bias parameters
        if Bias != None:
            self.params['params'] = Bias['params'];
        else:
            #Optimisation params - BIAS
            self.params = dict();
            self.params['targer_err'] = 0.00;
            if noTraining == False:
                if NGEN != None:
                    self.params['NGEN'] = NGEN;
                else:
                    self.params['NGEN'] = 100; #number of generations
            elif noTraining == True:
              self.params['NGEN'] = 1;

            ### NET CONFIG
            if netConfig != None:
                self.params['numI'] = netConfig['numI']; #number of input nodes
                self.params['numH'] = netConfig['numH']; #number of hidden nodes
                self.params['numO'] = netConfig['numO']; #number of output nodes
            else:
                self.params['numI'] = 2; #number of input nodes
                self.params['numH'] = 4; #number of hidden nodes
                self.params['numO'] = 1; #number of output nodes

            self.params['randomNodesInject'] = True; #DEFAULT
            self.params['temporalNodesInject'] = False; #temporal nodes injection - Switch ON before optim

            self.params['select_mut'] = 1;
            self.params['select_nextGen'] = 1;
            self.params['select_crossOver'] = 1;
            self.params['select_diffEvo'] = 1;
            if initConnectivity != None:
                self.params['initConnectivity'] = initConnectivity;
            else:
                self.params['initConnectivity'] = 'full'; #connectivity {'random', 'full'}
            if noIntraLayerConnHH != None:
                self.params['noIntraLayerConnHH']  = noIntraLayerConnHH;
            else:
                self.params['noIntraLayerConnHH'] = False; #no lateral connections between hidden units #TODO : update this to repo (3/7/14) - [Not Done ]
            if noContextLayer != None:
                self.params['noContextLayer'] = noContextLayer;
            else:
                self.params['noContextLayer'] = False; #no context layer for the hidden units #TODO : update this to repo (3/7/14) - [Not Done ]

            self.params['noNoise'] = False; # no gaussian noise for the hidden units #TODO : update this to repo (3/7/14) - [Not Done ]
            self.params['initTransferFn'] = 'all'; #random: randomly creates transfer functions, and doesn't account for duplication, #all: creates all possible combinations of transfer functions
            self.params['output_fns'] = [   outFn.identity_fn,
                                            outFn.sigmoid_fn,
                                            outFn.gaussian_fn,
                                            outFn.tanh_fn,
                                            outFn.arc_tanh_fn,
                                            outFn.gaussian_ii_fn
                                            ]; #Global output functions

            self.params['activation_fns'] = [inFn.max_dist,
                                             inFn.higher_order_prod,
                                             inFn.higher_order_sub,
                                             inFn.std_dev,
                                             inFn.mean,
                                             inFn.min ,
                                             inFn.max,
                                             inFn.inner_prod,
                                             inFn.euclidean_dist,
                                             inFn.manhattan_dist,
                                             inFn.max_dist
                                             ]; #Global input/weight/activaiton functions
            if hl_output_fns != None:
                self.params['hl_output_fns'] = hl_output_fns;
            else:
                self.params['hl_output_fns'] = [outFn.identity_fn,
                                            outFn.sigmoid_fn,
                                            outFn.gaussian_fn,
                                            outFn.tanh_fn,
                                            outFn.arc_tanh_fn,
                                            outFn.gaussian_ii_fn
                                           ]; #hidden layer output functions

            if ol_output_fns != None:
                self.params['ol_output_fns'] = ol_output_fns;
            else:
                self.params['ol_output_fns'] = [outFn.identity_fn,
                                            outFn.sigmoid_fn,
                                            outFn.gaussian_fn,
                                            outFn.tanh_fn,
                                            outFn.arc_tanh_fn,
                                            outFn.gaussian_ii_fn
                                            ]; #output layer output functions

            if hl_activation_fns != None:
                self.params['hl_activation_fns'] = hl_activation_fns;
            else:
                self.params['hl_activation_fns'] = [
                                                inFn.max_dist,
                                                inFn.higher_order_prod,
                                                inFn.higher_order_sub,
                                                inFn.std_dev,
                                                inFn.mean,
                                                inFn.min ,
                                                inFn.max,
                                                inFn.inner_prod,
                                                inFn.euclidean_dist,
                                                inFn.manhattan_dist


                                            ]; #hidden layer activation functions
            if ol_activation_fns != None:
                self.params['ol_activation_fns'] = ol_activation_fns;
            else:
                self.params['ol_activation_fns'] = [
                                                inFn.max_dist,
                                                inFn.higher_order_prod,
                                                inFn.higher_order_sub,
                                                inFn.std_dev,
                                                inFn.mean,
                                                inFn.min ,
                                                inFn.max,
                                                inFn.inner_prod,
                                                inFn.euclidean_dist,
                                                inFn.manhattan_dist

                                    ]; #output layer activation functions

            # Transfer function injection
            # Injection settings
            self.params['inject_sequence'] = ['projection', 'radial-basis', 'higher-order'];
            self.params['inject_trigger'] = 'interval'; # options 'stagnation', 'interval'
            self.params['temporal_change_interval'] = 25;
            self.params['curr_inject_sequence'] = 0;
            #- injection set
            self.params['hl_inject_act_fns'] = [];
            self.params['hl_inject_out_fns'] = [];
            self.params['ol_inject_act_fns'] = [];
            self.params['ol_inject_out_fns'] = [];

            ### Population size
            self.params['node_pop_size'] = 30;
            self.params['conn_pop_size'] = 30;
            self.params['model_pop_size'] = 30;
            self.params['prodConstantMax'] = 5;
            self.params['greedThreshold'] = 3; #For funnel search
            self.params['variation'] = 'varAnd'; #options : {varOr or varAnd} varOr - sample from (parent + child), varAnd samples from (child)
            self.params['percentNextGen'] = 0.5; #percentage of the population to select for next generation
            self.params['DEIterMax'] = 4; #max differential evolution iterations
            self.params['target_error'] = 0.0; # targer error
            self.params['prob_crossOver'] = 0.4; #probability of crossover
            self.params['prob_co_indp'] = 0.5; #probability of crossing over individual parameter
            self.params['prob_mutation'] = 0.6; #probablity of mutation
            self.params['prob_mut_indp'] = 0.2; #probability of mutating individual parameters
            self.params['alpha'] = 0.2; #significant of velocities for differential evolution
            self.params['param_min'] = -1.0; #parameter min
            self.params['param_max'] = 1.0; #parameter max
            self.params['do_diffEvolution'] = True; #does differential evolution
            self.params['do_crossOver'] = True; #cross over
            self.params['do_mutation'] = True; #mutation

            #Freeze component optimisation
            #-model
            self.params['optimFreeze_model'] = False;
            #-nodes
            self.params['optimFreeze_hidNodes'] = False;
            self.params['optimFreeze_outNodes'] = False;
            #-connectivity
            self.params['optimFreeze_connIH'] = True;
            self.params['optimFreeze_connHH'] = False;
            self.params['optimFreeze_connHO'] = True;
            #-weights
            self.params['optimFreeze_weightsIH'] = False;
            self.params['optimFreeze_weightsHH'] = False;
            self.params['optimFreeze_weightsHO'] = False;
            #mutation
            self.params['gaus_mut_mean'] = 0.1; #mutaion mean
            self.params['gaus_mut_std'] = 0.2; #mutation std. deviation
            #evolutionary algorithm
            self.params['tourn_size'] = 3;
            self.params['ea_ngen'] = 3;
            self.params['ea_mu_model'] = int(0.5 * self.params['model_pop_size']); #number of solutions to select for next generation
            self.params['ea_lamda_model'] = self.params['model_pop_size'] - self.params['ea_mu_model']; #pop to produce for next gen.
            self.params['ea_mu_nodes'] = int(0.5 * self.params['node_pop_size']); #number of solutions to select for next generation
            self.params['ea_lamda_nodes'] = self.params['node_pop_size'] - self.params['ea_mu_nodes']; #pop to produce for next gen.
            self.params['ea_mu_conn'] = int(0.5 * self.params['conn_pop_size']); #number of solutions to select for next generation
            self.params['ea_lamda_conn'] = self.params['conn_pop_size'] - self.params['ea_mu_conn']; #pop to produce for next gen.

        #TOOLBOX
        self.tbox = ndmCoevoBase.init_toolbox(self.params);
        self.varTbox = coevoOptim.getToolbox(self.params);

        #NO TRAINING
        if noTraining:
            self.params['optimFreeze_model'] = True;
            #-nodes
            self.params['optimFreeze_hidNodes'] = True;
            self.params['optimFreeze_outNodes'] = True;
            #-connectivity
            self.params['optimFreeze_connIH'] = True;
            self.params['optimFreeze_connHH'] = True;
            self.params['optimFreeze_connHO'] = True;
            #-weights
            self.params['optimFreeze_weightsIH'] = True;
            self.params['optimFreeze_weightsHH'] = True;
            self.params['optimFreeze_weightsHO'] = True;

    def gen_random_nodes(self,layer, n = 5):
        """
        generate random nodes with random transfer functions
        """

        rnd_nodes = [self.tbox.node() for _ in xrange(n)];
        if layer == 'hidden':
            ndmCoevoBase.init_tf_fn_forPop(rnd_nodes , params =self.params, layer = 'hidden');
        elif layer == 'output':
            ndmCoevoBase.init_tf_fn_forPop(rnd_nodes, params = self.params, layer = 'output');
        else:
            raise Exception("Invalid layer specification in gen_random_nodes()");

        return rnd_nodes;

    def gen_temporal_nodes(self,layer, n = 5):
        """
        generate random nodes from the current temporal pattern
        """

        #create the new nodes
        tmp_nodes = [self.tbox.node() for _ in xrange(n)];

        #get the current pattern for the current tempo.
        fn_class = None;
        if self.current_iter % self.params['temporal_change_interval'] == 0:


            #increament and change
            if self.params['curr_inject_sequence'] < len(self.params['inject_sequence'])-1:
                #change then increament
                self.params['curr_inject_sequence'] +=1
            else:
                #start from begining
                self.params['curr_inject_sequence'] = 0;


            #change
            fn_class = self.params['inject_sequence'][self.params['curr_inject_sequence']];

            #verbose
            print ">>fn_class", fn_class;

            #switch transfer function set used for optimisation
            self.params['hl_activation_fns'] = inFn.getFunctionsClass(fn_class);
            self.params['hl_output_fns_fns'] = outFn.getFunctionsClass(fn_class);
            self.params['ol_activation_fns'] = inFn.getFunctionsClass(fn_class);
            self.params['ol_output_fns'] = outFn.getFunctionsClass(fn_class);

        #set their transfer functions
        if layer == 'hidden':
            ndmCoevoBase.init_tf_fn_forPop(tmp_nodes , params = self.params, layer = 'hidden');
        elif layer == 'output':
            ndmCoevoBase.init_tf_fn_forPop(tmp_nodes,  params =self.params, layer = 'output');
        else:
            raise Exception("Invalid layer specification in gen_temporal_nodes()");

        return tmp_nodes;

    def create_model(self, components):
        """
        Create a model from the components
        """
        sol = dict();

        #prepare components
                            #model
        sol['model'] = components['model'];
                            #connections
        sol['connActive_IH'] = commons.matrixForm(components['connActive_IH'],self.params['numI'], self.params['numH']);
        sol['connActive_HH'] = commons.matrixForm(components['connActive_HH'],self.params['numH'], self.params['numH']);
        sol['connActive_HO'] = commons.matrixForm(components['connActive_HO'],self.params['numH'], self.params['numO']);
                            #weights
        sol['connWeights_IH'] = commons.matrixForm(components['connWeights_IH'],self.params['numI'], self.params['numH']);
        sol['connWeights_HH'] = commons.matrixForm(components['connWeights_HH'],self.params['numH'], self.params['numH']);
        sol['connWeights_HO'] = commons.matrixForm(components['connWeights_HO'],self.params['numH'], self.params['numO']);
                            #unpack nodes
        sol['hidden_nodes'] = [node for nkey,node in components['hidden_nodes'].iteritems() ];
        sol['out_nodes'] = [node for nkey,node in components['out_nodes'].iteritems() ];

        try:
            #create model
            model = ndmModel(numI = self.params['numI'],
                             numH =  self.params['numH'],
                             numO = self.params['numO'],
                             components = sol,
                             noIntraLayerConnHH = self.params['noIntraLayerConnHH'],
                             noContextLayer = self.params['noContextLayer'],
                             noNoise = self.params['noNoise']); #TODO : update this to repo (3/7/14) - [Not Done ]

        except AttributeError:
            raise Exception("Oopps - something went wrong while evaluating");

        return model;

    def evaluate_mpi(self,components,verbose = False, copy = True, eval_type = None, **kwargs):
        """
        Evaluates a given set of components frin all species in parallel
        """

          #fitnesses
        fitness_test = 1;
        fitness_val = 1;


        if copy:
            sol_eval = self.tbox.clone(components);
        else:
            sol_eval = components;

        try:

            #Make model
            model_eval = self.create_model(sol_eval);
            #evaluate on training set
            fitness_train  = model_eval.evaluate(self.train_set);

            if eval_type != None:

                ### EVALUATE ON TEST SET
                if eval_type == 'test':
                    fitness_test  = model_eval.evaluate(self.test_set);


                ### EVALUATE ON VALIDATION SET
                if eval_type == 'validation':
                    fitness_val = model_eval.evaluate(self.validation_set);


        except AttributeError:
            raise Exception("Oopps - something went wrong while evaluating");

        return fitness_train,fitness_val,fitness_test;

    def evaluate(self, components, verbose = False, copy = True, eval_type = None,**kwargs):
        """
        Evaluate a given solution of components from all species
        """

        #fitnesses
        fitness_test = 1;
        fitness_val = 1;


        if copy:
            sol_eval = self.tbox.clone(components);
        else:
            sol_eval = components;

        try:

            #Make model
            model_eval = self.create_model(sol_eval);
            #evaluate on training set
            fitness_train  = model_eval.evaluate(self.train_set);

            if eval_type != None:

                ### EVALUATE ON TEST SET
                if eval_type == 'test':
                    fitness_test  = model_eval.evaluate(self.test_set);


                ### EVALUATE ON VALIDATION SET
                if eval_type == 'validation':
                    fitness_val = model_eval.evaluate(self.validation_set);


        except AttributeError:
            raise Exception("Oopps - something went wrong while evaluating");

        #stor best cost
        if fitness_train < self.best_cost:
            #store best cost
            self.best_cost = fitness_train;
            print "Best_cost:", self.best_cost;

            #store solutions every couple of generations
            self.best_sol = (sol_eval);
            self.best_model = self.create_model(sol_eval);

        if fitness_val < self.best_cost_val:
            #store the best cots (validation)
            self.best_cost_val = fitness_val;
            print "Best_cost(validation)", self.best_cost_val;

            self.best_sol_val = sol_eval;
            self.best_model_val = self.create_model(sol_eval);

        if fitness_test < self.best_cost_test:
            #store the best cost (test)
            self.best_cost_test = fitness_test;
            print "Best_cost(test)", self.best_cost_test;
            self.best_sol_test = sol_eval;
            self.best_model_test = self.create_model(sol_eval);


        return fitness_train,fitness_val,fitness_test;


    def evaluate_mpi(self, components, verbose = False, copy = True, eval_type = None,**kwargs):
        """
        Evaluate a given solution of components from all species
        """

        #fitnesses
        fitness_test = 1;
        fitness_val = 1;


        if copy:
            sol_eval = self.tbox.clone(components);
        else:
            sol_eval = components;

        try:

            #Make model
            model_eval = self.create_model(sol_eval);
            #evaluate on training set
            fitness_train  = model_eval.evaluate(self.train_set);

            if eval_type != None:

                ### EVALUATE ON TEST SET
                if eval_type == 'test':
                    fitness_test  = model_eval.evaluate(self.test_set);


                ### EVALUATE ON VALIDATION SET
                if eval_type == 'validation':
                    fitness_val = model_eval.evaluate(self.validation_set);


        except AttributeError:
            raise Exception("Oopps - something went wrong while evaluating");

        #stor best cost
        if fitness_train < self.best_cost:
            #store best cost
            self.best_cost = fitness_train;
            print "Best_cost:", self.best_cost;

            #store solutions every couple of generations
            self.best_sol = (sol_eval);
            self.best_model = self.create_model(sol_eval);

        if fitness_val < self.best_cost_val:
            #store the best cots (validation)
            self.best_cost_val = fitness_val;
            print "Best_cost(validation)", self.best_cost_val;

            self.best_sol_val = sol_eval;
            self.best_model_val = self.create_model(sol_eval);

        if fitness_test < self.best_cost_test:
            #store the best cost (test)
            self.best_cost_test = fitness_test;
            print "Best_cost(test)", self.best_cost_test;
            self.best_sol_test = sol_eval;
            self.best_model_test = self.create_model(sol_eval);


        return fitness_train,fitness_val,fitness_test;

    def evaluate_connIH_map(self,connIH):
        """
        evaluates an input-hidden connection topology in parallel using dtm - (MPI)
        """
        sol = self.tbox.clone(self.represntatives);

        if connIH.fitness.valid != True:
            sol['connActive_IH'] = connIH;
            f = self.evaluate(sol,copy = False, eval_type= 'validation');
            connIH.fitness.values = f[0],f[1];

        del sol;

        return connIH;

    def evaluate_connIH(self, next_gen):
        """
        This method unlike the evaluate method evaluates a single component
        """

        sol = self.tbox.clone(self.represntatives);

        for connIH in next_gen:
            if connIH.fitness.valid != True:
                sol['connActive_IH'] = connIH;
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                connIH.fitness.values = f[0],f[1];


                # #Save best cost
                # last = len(self.snapshot['topology']) -1;
                # if f[0] < self.snapshot['topology'][last]:
                #     self.snapshot['topology'].append(f[0]);

        del sol;


    def evaluate_connHH_map(self,connHH):
        """
        evaluate hidden to hidden connections in parallel
        """

        sol = self.tbox.clone(self.represntatives);

        if connHH.fitness.valid != True:
            sol['connActive_HH'] = connHH;
            f = self.evaluate(sol,copy = False, eval_type= 'validation');
            connHH.fitness.values = f[0],f[1];


        del sol;

        return connHH;

    def evaluate_connHH(self, next_gen):
        """
        evaluate hidden to hidden connections
        """

        sol = self.tbox.clone(self.represntatives);

        for connHH in next_gen:
            if connHH.fitness.valid != True:
                sol['connActive_HH'] = connHH;
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                connHH.fitness.values = f[0],f[1];


                # #Save best cost
                # last = len(self.snapshot['topology']) -1;
                # if f[0] < self.snapshot['topology'][last]:
                #     self.snapshot['topology'].append(f[0]);


        del sol;


    def evaluate_connHO_map(self,connHO):
        """
         parallel evaluation of the hidden to output units
        """
        sol = self.tbox.clone(self.represntatives);

        if connHO.fitness.valid != True:
            sol['connActive_HO'] = connHO;
            f = self.evaluate(sol,copy = False, eval_type= 'validation');
            connHO.fitness.values = f[0],f[1];


        del sol;

        return connHO;

    def evaluate_connHO(self, next_gen):
        """
        connHO
        """
        sol = self.tbox.clone(self.represntatives);

        for connHO in next_gen:
            sol['connActive_HO'] = connHO;
            f = self.evaluate(sol,copy = False, eval_type= 'validation');
            connHO.fitness.values = f[0],f[1];


        del sol;


    def evaluate_weightIH_map(self,weightIH):
        """
        evaluates in parallel
        """
        sol = self.tbox.clone(self.represntatives);

        if weightIH.fitness.valid != True:
                sol['connWeights_IH'] = weightIH;
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                weightIH.fitness.values = f[0],[1];


        #free up memory
        del sol;

        return weightIH;


    def evaluate_weightIH(self,next_gen):
        """
        evaluates a given set of weights
        """
        #make a copy
        sol = self.tbox.clone(self.represntatives);

        for weightIH in next_gen:
            if weightIH.fitness.valid != True:
                sol['connWeights_IH'] = weightIH;
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                weightIH.fitness.values = f[0],[1];

                # #Save best cost
                # last = len(self.snapshot['weights']) -1;
                # if f[0] < self.snapshot['weights'][last]:
                #     self.snapshot['weights'].append(f[0]);

        #free up memory
        del sol;

    def evaluate_weightHH_map(self, weightHH):
        """
        """

        sol = self.tbox.clone(self.represntatives);

        if weightHH.fitness.valid != True:
                sol['connWeights_HH'] = weightHH;
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                weightHH.fitness.values = f[0],f[1];

        #free up memory
        del sol;

        return weightHH;

    def evaluate_weightHH(self,next_gen):
        """
        evaluates the lateral connections
        """

        #make a copy
        sol = self.tbox.clone(self.represntatives);

        for weightHH in next_gen:
            if weightHH.fitness.valid != True:
                sol['connWeights_HH'] = weightHH;
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                weightHH.fitness.values = f[0],f[1];

                # #Save best cost
                # last = len(self.snapshot['weights']) -1;
                # if f[0] < self.snapshot['weights'][last]:
                #     self.snapshot['weights'].append(f[0]);

        #free up memory
        del sol;

    def evaluate_weightHO_mpi(self, weightHO, comm):
        """

        Evalutes the candidates of weights between the Hidden output layer
        """

        sol = self.tbox.clone(self.represntatives);

        if weightHO.fitness.valid != True:
            #replace
            sol['connWeights_HO'] = weightHO;
            #evaluate
            f = self.evaluate(sol,copy = False, eval_type= 'validation');
            weightHO.fitness.values = f[0],f[1];

            if comm.rank == 0:
                #Save best cost
                last = len(self.snapshot['weights']) -1;
                if f[0] < self.snapshot['weights'][last]:
                    self.snapshot['weights'].append(f[0]);

        del sol;


    def evaluate_weightHO_map(self, weightsHO):
        """
        evaluates weights from hidden to output
        """

        sol = self.tbox.clone(self.represntatives);


        if weightsHO.fitness.valid != True:
            #replace
            sol['connWeights_HO'] = weightsHO;
            #evaluate
            f = self.evaluate(sol,copy = False, eval_type= 'validation');
            weightsHO.fitness.values = f[0],f[1];


        del sol;

        return  weightsHO;


    def evaluate_weightHO(self, next_gen):
        """
        evaluates weights from hidden to output
        """

        sol = self.tbox.clone(self.represntatives);

        for weightsHO in next_gen:
            if weightsHO.fitness.valid != True:
                #replace
                sol['connWeights_HO'] = weightsHO;
                #evaluate
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                weightsHO.fitness.values = f[0],f[1];


                # #Save best cost
                # last = len(self.snapshot['weights']) -1;
                # if f[0] < self.snapshot['weights'][last]:
                #     self.snapshot['weights'].append(f[0]);

        del sol;

    def evaluate_model_map(self, model):
        """
        Evaluate models in parallel
        """
        sol = self.tbox.clone(self.represntatives);

        if model.fitness.valid != True:
                sol['model'] = model;
                #evaluate the model
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                model.fitness.values = f[0],f[1];

        #GC
        del sol;

        return model;




    def evaluate_model(self,next_gen):
        """
        evaluates a given model
        """

        #make a copy
        sol = self.tbox.clone(self.represntatives);

        #evaluate all models
        for model in next_gen:
            if model.fitness.valid != True:
                sol['model'] = model;
                #evaluate the model
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                model.fitness.values = f[0],f[1];

                # #Save best cost
                # last = len(self.snapshot['model']) -1;
                # if f[0] < self.snapshot['model'][last]:
                #     self.snapshot['model'].append(f[0]);

        #GC
        del sol;


    def evaluate_hiddenNodes_map(self,h_node, hi):
        """
        implementation that works using mpi for parallel computing
        """

        sol = self.tbox.clone(self.represntatives);

        if h_node.fitness.valid != True:
                #replace
                sol['hidden_nodes'][hi] = h_node;
                #evaluate
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                h_node.fitness.values = f[0],f[1]


        #GC
        del sol;
        return h_node;

    def evaluate_hiddenNodes(self,next_gen,hi):
        """
        evaluates a hidden node
        """
        sol = self.tbox.clone(self.represntatives);

        for h_node in next_gen:

            if h_node.fitness.valid != True:
                #replace
                sol['hidden_nodes'][hi] = h_node;
                #evaluate
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                h_node.fitness.values = f[0],f[1];

                # #Save best cost
                # last = len(self.snapshot['hidden_layer']) -1;
                # if f[0] < self.snapshot['hidden_layer'][last]:
                #     self.snapshot['hidden_layer'].append(f[0]);

        #GC
        del sol;


    def evaluate_outputNodes_map(self, o_node, oi):
        """
        """

        sol = self.tbox.clone(self.represntatives);

        if o_node.fitness.valid != True:
                #replace
                sol['out_nodes'][oi] = o_node;
                #evaluate
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                o_node.fitness.values = f[0],f[1];


        #GC
        del sol;

        return o_node;


    def evaluate_outputNodes(self, next_gen, oi):
        """
        evaluate output nodes
        """

        sol = self.tbox.clone(self.represntatives);

        for o_node in next_gen:

            if o_node.fitness.valid != True:
                #replace
                sol['out_nodes'][oi] = o_node;
                #evaluate
                f = self.evaluate(sol,copy = False, eval_type= 'validation');
                o_node.fitness.values = f[0],f[1];

                #Save best cost
                # last = len(self.snapshot['output_layer']) -1;
                # if f[0] < self.snapshot['output_layer'][last]:
                #     self.snapshot['output_layer'].append(f[0]);


        #GC
        del sol;


    def init_populations(self):
        """
        Intialises the populations based on the parameter settings
        """

        #hidden nodes
        for ni in xrange(self.params['numH']):
            if self.params['initTransferFn'] == 'random':
                self.pop['hidden_nodes'][str(ni)] = [self.tbox.node() for _ in xrange(self.params['node_pop_size'])];
            elif self.params['initTransferFn'] == 'all':
                tfs_comb =len([(w,n) for w in self.params['hl_activation_fns'] for n in self.params['hl_output_fns']]);
                self.pop['hidden_nodes'][str(ni)] = [self.tbox.node() for _ in xrange(tfs_comb)];

        #output nodes
        for ni in xrange(self.params['numO']):

            if self.params['initTransferFn'] == 'random':
                self.pop['out_nodes'][str(ni)] = [self.tbox.node() for _ in xrange(self.params['node_pop_size'])];
            elif self.params['initTransferFn'] == 'all':
                tfs_comb =len([(w,n) for w in self.params['ol_activation_fns'] for n in self.params['ol_output_fns']]);
                self.pop['out_nodes'][str(ni)] = [self.tbox.node() for _ in xrange(tfs_comb)];

        #initialise transfer functions
        if self.params['initTransferFn'] == 'all':
            ndmCoevoBase.init_all_transferFns(self.pop['hidden_nodes'], params = self.params, layer = 'hidden');
            ndmCoevoBase.init_all_transferFns(self.pop['out_nodes'], params = self.params, layer = 'output');
        elif self.params['initTransferFn'] == 'random':
            ndmCoevoBase.init_tf_fn_random(self.pop['hidden_nodes'], params =self.params, layer = 'hidden');
            ndmCoevoBase.init_tf_fn_random(self.pop['out_nodes'], params = self.params, layer = 'output');
        else:
            raise Exception("No such initTransferFn method, use 'all' or 'random'");


        #connectivity/toplogy
        self.pop['connActive_IH_pop'] = [self.tbox.connActive_IH() for _ in xrange(self.params['conn_pop_size'])];
        self.pop['connActive_HH_pop'] = [self.tbox.connActive_HH() for _ in xrange(self.params['conn_pop_size'])];
        self.pop['connActive_HO_pop'] = [self.tbox.connActive_HO() for _ in xrange(self.params['conn_pop_size'])];
        #weights
        self.pop['connWeights_IH_pop'] = [self.tbox.connWeights_IH() for _ in xrange(self.params['conn_pop_size'])];
        self.pop['connWeights_HH_pop'] = [self.tbox.connWeights_HH() for _ in xrange(self.params['conn_pop_size'])];
        self.pop['connWeights_HO_pop'] = [self.tbox.connWeights_HO() for _ in xrange(self.params['conn_pop_size'])];
        #models
        self.pop['model_pop'] = [self.tbox.model() for _ in xrange(self.params['model_pop_size'])];


    def init_empty_population(self):
        """
        Initialise and empty population
        """

        #hidden nodes
        for ni in xrange(self.params['numH']):
            self.pop['hidden_nodes'][str(ni)] = [];

        #output nodes
        for ni in xrange(self.params['numO']):
            self.pop['out_nodes'][str(ni)] = [];


        #connectivity/toplogy
        self.pop['connActive_IH_pop'] = [];
        self.pop['connActive_HH_pop'] = [];
        self.pop['connActive_HO_pop'] = [];
        #weights
        self.pop['connWeights_IH_pop'] = [];
        self.pop['connWeights_HH_pop'] = [];
        self.pop['connWeights_HO_pop'] = [];
        #models
        self.pop['model_pop'] = [];




    def select_representative(self):
        """
        Returns a dictionary of a random set of representatives from all the species
        """
        #Select representatives randomly
        representative = dict();
        representative['model'] = tools.selRandom(self.pop['model_pop'],1)[0];
        representative['connActive_IH'] = tools.selRandom(self.pop['connActive_IH_pop'],1)[0];
        representative['connActive_HH'] = tools.selRandom(self.pop['connActive_HH_pop'],1)[0];
        representative['connActive_HO'] = tools.selRandom(self.pop['connActive_HO_pop'],1)[0];
        representative['connWeights_IH'] = tools.selRandom(self.pop['connWeights_IH_pop'],1)[0];
        representative['connWeights_HH'] = tools.selRandom(self.pop['connWeights_HH_pop'],1)[0];
        representative['connWeights_HO'] = tools.selRandom(self.pop['connWeights_HO_pop'],1)[0];

        representative['hidden_nodes'] = dict();
        for k,v in self.pop['hidden_nodes'].iteritems():
            representative['hidden_nodes'][str(k)] = tools.selRandom(v,1)[0];

        representative['out_nodes'] = dict();
        for k,v in self.pop['out_nodes'].iteritems():
            representative['out_nodes'][str(k)] = tools.selRandom(v,1)[0];


        return representative;




    def coevolve(self):
        """
        Optimise neural network architectures
        """

        # #set plotter
        # plt.ion();
        # plt.show();
        # plt.title("Error");
        # plt.xlabel("Generations(t)");
        # plt.ylabel("Error(MSE)");
        # plt.ylim([0.0,1.0]);

        #select representative
        self.represntatives = self.select_representative();
        #next representative should clone the current representative components
        next_repr = self.tbox.clone(self.represntatives);


        ## PRINT OUT SOME OPTIMISATION PARAMETERS - FOR CONFIRMATION #TODO : update this to repo (3/7/14) - [Not Done ]
        print " ################ Optimisation Params ################"
        for param, value in self.params.iteritems():
            print "> param: ", param, " : ", value;

        #Coevolution Loop
        for self.current_iter in xrange(self.params['NGEN']):


            #Check stopping criteria
            # if round(self.best_cost,2) < self.params['targer_err']:
            #     #STOP THE OPTIMISATION
            #     break;

            print "GEN", self.current_iter;


            #Iterate over species
            for s_key, specie in self.pop.iteritems():

                #Use representatives to evaluate the individual in the current specie
                if s_key == 'hidden_nodes':

                        #for all the hidden nodes positions
                        for hi,h_nodes in specie.iteritems():
                            #evaluate candidates for that pos.
                            for i,n in enumerate(h_nodes):
                                #clone
                                sol = self.tbox.clone(self.represntatives);
                                #replace node hi
                                sol['hidden_nodes'][hi] = n;
                                #evaluate
                                f = self.evaluate(sol,copy = False);
                                n.fitness.values = f[0], f[1];

                                if debug:
                                    print "n act. fn", n.activation_fn;
                                    print "n out. fn", n.output_fn;
                                    print "hidden node",hi,"fitness:", n.fitness.values[0];

                                #delete
                                del sol;


                            #should optimise?
                            next_gen = None;
                            if not self.params['optimFreeze_hidNodes']:
                                print "-optim hidNodes";
                                #Variations (Cross over and Mutation)
                                if self.params['variation'] == 'varOr': #variate and select from  (parent + offsprings)
                                    next_gen = algorithms.varOr(population = self.pop['hidden_nodes'][hi],
                                                            toolbox = self.varTbox['real'],
                                                            lambda_  = self.params['node_pop_size'],
                                                            cxpb = 1 - self.params['prob_mutation'],
                                                            mutpb = self.params['prob_mutation']);
                                elif self.params['variation'] == 'varAnd':
                                    next_gen = algorithms.varAnd(population = self.pop['hidden_nodes'][hi],
                                                                 toolbox = self.varTbox['real'],
                                                                 cxpb = self.params['prob_crossOver'],
                                                                 mutpb = self.params['prob_mutation']
                                                                 );

                                #generate random nodes
                                if self.params['randomNodesInject']:
                                    print ">>inject";
                                    rand_nodes = self.gen_random_nodes(layer = 'hidden',n = 10);
                                    #add to next gen
                                    next_gen.extend(rand_nodes);

                                #generate temporal nodes
                                if self.params['temporalNodesInject']:
                                    print ">>Inject - Temporal patterning";
                                    temp_nodes = self.gen_temporal_nodes(layer = 'hidden', n = 10);
                                    #add to next gen
                                    next_gen.extend(temp_nodes);


                                #Evaluate
                                self.evaluate_hiddenNodes(next_gen,hi);



                                #next generation
                                if next_gen == None:
                                    next_gen = self.pop['hidden_nodes'][hi];

                                #selection of next population
                                self.pop['hidden_nodes'][hi] = tools.selTournament(next_gen,
                                                                            self.params['node_pop_size'],
                                                                            self.params['tourn_size']);

                            #selection of next representative
                            best_hidden_node = tools.selBest(self.pop['hidden_nodes'][hi],1)[0];
                            next_repr['hidden_nodes'][hi] = self.tbox.clone(best_hidden_node);



                ##################OUTPUT NODES ######################
                if s_key == 'out_nodes':
                        #for all the output nodes
                        for oi,o_nodes in specie.iteritems():
                            #evaluate candidate members for that output node position
                            for i,n in enumerate(o_nodes):
                                #clone
                                sol = self.tbox.clone(self.represntatives);
                                #replace node ni
                                sol['out_nodes'][oi] = n;

                                #evaluate
                                f = self.evaluate(sol,copy = False);
                                n.fitness.values = f[0],f[1];

                                if debug:
                                    print "output node",oi,"fitness:", n.fitness.values[0];

                                #delete
                                del sol;



                            #should optimise?
                            next_gen = None;
                            if not self.params['optimFreeze_outNodes']:
                                print "-optim outNodes";
                                #Variations (Cross over and Mutation)
                                if self.params['variation'] == 'varOr': #variate and select from  (parent + offsprings)
                                    next_gen = algorithms.varOr(population = self.pop['out_nodes'][oi],
                                                            toolbox = self.varTbox['real'],
                                                            lambda_  = self.params['node_pop_size'],
                                                            cxpb = 1 - self.params['prob_mutation'],
                                                            mutpb = self.params['prob_mutation']);
                                elif self.params['variation'] == 'varAnd':
                                    next_gen = algorithms.varAnd(population = self.pop['out_nodes'][oi],
                                                                 toolbox = self.varTbox['real'],
                                                                 cxpb = self.params['prob_crossOver'],
                                                                 mutpb = self.params['prob_mutation']
                                                                 );
                                #generate random nodes
                                if self.params['randomNodesInject']:
                                    print ">>inject";
                                    rand_nodes = self.gen_random_nodes(layer = 'output',n = 10);
                                    #add to next gen
                                    next_gen.extend(rand_nodes);

                                #generate temporal nodes
                                if self.params['temporalNodesInject']:
                                    print ">>Inject - Temporal patterning";
                                    temp_nodes = self.gen_temporal_nodes(layer = 'output', n = 10);
                                    #add to next gen
                                    next_gen.extend(temp_nodes);

                                #Evaluate
                                self.evaluate_outputNodes(next_gen,oi);


                                #next generation
                                if next_gen == None:
                                    next_gen = self.pop['out_nodes'][oi];

                                #selection of next population
                                self.pop['out_nodes'][oi] = tools.selTournament(next_gen,
                                                                            self.params['node_pop_size'],
                                                                            self.params['tourn_size']);

                            #selection of next representative
                            next_repr['out_nodes'][oi] = self.tbox.clone(tools.selBest(self.pop['out_nodes'][oi],1)[0]);


                ################## MODEL  ######################
                if s_key == 'model_pop':
                        #evaluate the models
                        for model in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['model'] = model;

                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            model.fitness.values = f[0],f[1];

                            if debug:
                                print "model fitness:", model.fitness.values[0];
                            del sol;


                        #should we optimise?
                        next_gen = None;
                        if not self.params['optimFreeze_model']:
                            print "-optim model";
                            if self.params['variation'] == 'varAnd':
                                #Variations (Cross over and Mutation)
                                next_gen = algorithms.varAnd(self.pop['model_pop'],
                                                                            self.varTbox['binary'],
                                                                            self.params['prob_crossOver'],
                                                                            self.params['prob_mutation']
                                );
                            elif self.params['variation'] == 'varOr':
                                next_gen= algorithms.varOr(population = self.pop['model_pop'],
                                                                        toolbox =    self.varTbox['binary'],
                                                                        lambda_ = self.params['model_pop_size'],
                                                                        cxpb =  1- self.params['prob_mutation'],
                                                                        mutpb=   self.params['prob_mutation']
                                );

                            #evaluate
                            self.evaluate_model(next_gen);

                            #selection of next population
                            if next_gen == None:
                                next_gen = self.pop['model_pop'];
                            #select from next generation
                            self.pop['model_pop'] = tools.selTournament(next_gen,
                                                                        self.params['model_pop_size'],
                                                                        self.params['tourn_size']);

                        #selection of next population
                        next_repr['model'] = self.tbox.clone(tools.selBest(self.pop['model_pop'],1)[0]);

                ##################CONN_ACTIVE INPUT TO HIDDEN LAYER ######################
                if s_key == 'connActive_IH_pop':
                        for conn in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['connActive_IH'] =conn;

                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            conn.fitness.values = f[0], f[1];

                            if debug:
                                print "connActiveIH fitness:", conn.fitness.values[0];
                            del sol;


                        #next generation
                        next_gen = None;
                        if not self.params['optimFreeze_connIH']:
                            print "-optim connIH";
                            #Variations (Cross over and Mutation)
                            if self.params['variation'] =='varOr': #select (parents + children)
                                next_gen = algorithms.varOr(population =self.pop['connActive_IH_pop'],
                                                                                    toolbox = self.varTbox['binary'],
                                                                                    lambda_ = self.params['conn_pop_size'],
                                                                                     cxpb = 1- self.params['prob_mutation'],
                                                                                     mutpb = self.params['prob_mutation']
                                 );
                            elif self.params['variation'] == 'varAnd':
                                next_gen = algorithms.varAnd(population = self.pop['connActive_IH_pop'],
                                                                                  toolbox = self.varTbox['binary'],
                                                                                  cxpb  = self.params['prob_crossOver'],
                                                                                  mutpb = self.params['prob_mutation']);


                            #evaluate
                            #TODO

                            #next generation
                            if next_gen == None:
                                next_gen = self.pop['connActive_IH_pop'];
                            #selection of next population
                            self.pop['connActive_IH_pop'] = tools.selTournament(next_gen,
                                                                         self.params['conn_pop_size'],
                                                                         self.params['tourn_size']);
                        #selection of next representative
                        next_repr['connActive_IH'] = self.tbox.clone(tools.selBest(self.pop['connActive_IH_pop'],1)[0]);


                ##################CONN_ACTIVE HIDDEN TO HIDDEN LAYER ######################
                if s_key == 'connActive_HH_pop' \
                            and self.params['numH'] > 1:
                        for conn in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['connActive_HH'] = conn;

                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            conn.fitness.values = f[0], f[1];
                            if debug:
                                print "connActiveHH fitness:", conn.fitness.values[0];
                            del sol;



                        #should freeze optimisation?
                        next_gen = None;
                        if not self.params['optimFreeze_connHH']:
                            print "-optim connHH";
                            #Variations (Cross over and Mutation)
                            if self.params['variation'] =='varOr': #select (parents + children)

                                next_gen = algorithms.varOr(population =self.pop['connActive_HH_pop'],
                                                                                    toolbox = self.varTbox['binary'],
                                                                                    lambda_ = self.params['conn_pop_size'],
                                                                                     cxpb = 1-self.params['prob_mutation'],
                                                                                     mutpb = self.params['prob_mutation']
                                );
                            elif self.params['variation'] == 'varAnd':
                                next_gen = algorithms.varAnd(population = self.pop['connActive_HH_pop'],
                                                                                  toolbox = self.varTbox['binary'],
                                                                                  cxpb  = self.params['prob_crossOver'],
                                                                                  mutpb = self.params['prob_mutation']);

                            #evaluate
                            self.evaluate_connHH(next_gen);


                            #next generation
                            if next_gen == None:
                                next_gen = self.pop['connWeights_HH_pop'];
                            #selection of next population
                            self.pop['connActive_HH_pop'] = tools.selTournament(next_gen,
                                                                         self.params['conn_pop_size'],
                                                                         self.params['tourn_size']);

                        #selection of next population
                        next_repr['connActive_HH'] = self.tbox.clone(tools.selBest(self.pop['connActive_HH_pop'],1)[0]);

                ##################CONN_ACTIVE HIDDEN TO OUTPUT LAYER ######################
                if s_key == 'connActive_HO_pop':
                        for conn in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['connActive_HO'] = conn;
                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            conn.fitness.values = f[0], f[1];
                            if debug:
                                print "connActiveHO fitness:", conn.fitness.values[0];
                            del sol;



                        #should freeze optimisation?
                        next_gen = None;
                        if not self.params['optimFreeze_connHO']:
                            print "-optim connHO";
                            #Variations (Cross over and Mutation)
                            if self.params['variation'] =='varOr': #select (parents + children)
                                next_gen = algorithms.varOr(population =self.pop['connActive_HO_pop'],
                                                                                    toolbox = self.varTbox['binary'],
                                                                                    lambda_ = self.params['conn_pop_size'],
                                                                                     cxpb = 1-self.params['prob_mutation'],
                                                                                     mutpb = self.params['prob_mutation']
                                 );
                            elif self.params['variation'] == 'varAnd':
                                next_gen= algorithms.varAnd(population = self.pop['connActive_HO_pop'],
                                                                                  toolbox = self.varTbox['binary'],
                                                                                  cxpb  = self.params['prob_crossOver'],
                                                                                  mutpb = self.params['prob_mutation']);


                            #evaluate
                            # if
                            # self.evaluate_connHO(next_gen);


                            if next_gen == None:
                                next_gen = self.pop['connActive_HO_pop'];

                            #selection of next population
                            self.pop['connActive_HO_pop'] = tools.selTournament(next_gen,
                                                                         self.params['conn_pop_size'],
                                                                         self.params['tourn_size']);

                        #selection of next population
                        next_repr['connActive_HO'] = self.tbox.clone(tools.selBest(self.pop['connActive_HO_pop'],1)[0]);

                ##################CONN_WEIGHTS INPUT TO HIDDEN LAYER ######################
                if s_key == 'connWeights_IH_pop':
                        for weights in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['connWeights_IH'] = weights;
                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            weights.fitness.values = f[0],f[1];
                            if debug:
                                print "connWeightsIH fitness:", weights.fitness.values[0];
                            del sol;


                        #should freeze optimisation?
                        next_gen = None;
                        if not self.params['optimFreeze_weightsIH']:
                            print "-optim weightsIH";
                            #Variations (Cross over and Mutation)
                            if self.params['variation'] =='varOr': #select (parents + children)
                                    next_gen = algorithms.varOr(population =self.pop['connWeights_IH_pop'],
                                                                                        toolbox = self.varTbox['real'],
                                                                                        lambda_ = self.params['conn_pop_size'],
                                                                                         cxpb = 1-self.params['prob_mutation'],
                                                                                         mutpb = self.params['prob_mutation']
                                    );
                            elif self.params['variation'] == 'varAnd':
                                    next_gen = algorithms.varAnd(population = self.pop['connWeights_IH_pop'],
                                                                                      toolbox = self.varTbox['real'],
                                                                                      cxpb  = self.params['prob_crossOver'],
                                                                                      mutpb = self.params['prob_mutation']);

                            #evaluate
                            self.evaluate_weightIH(next_gen);


                            #next generation
                            if next_gen == None:
                                next_gen = self.pop['connWeights_IH_pop'];

                            #selection of next population
                            self.pop['connWeights_IH_pop'] = tools.selTournament(next_gen,
                                                                         self.params['conn_pop_size'],
                                                                         self.params['tourn_size']);

                        #selection of next population
                        next_repr['connWeights_IH'] = self.tbox.clone(tools.selBest(self.pop['connWeights_IH_pop'],1)[0]);


                ##################CONN_WEIGHTS HIDDEN TO HIDDEN LAYER ######################
                if s_key == 'connWeights_HH_pop' \
                            and self.params['numH']>1:
                        for weights in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['connWeights_HH'] = weights;
                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            weights.fitness.values = f[0],f[1];
                            if debug:
                                print "connWeightsHH fitness:", weights.fitness.values[0];
                            del sol;



                        #should freeze optimisation?
                        next_gen = None;
                        if not self.params['optimFreeze_weightsHH']:
                                print "-optim weightsHH";

                                #Variations (Cross over and Mutation)
                                if self.params['variation'] =='varOr': #select (parents + children)
                                        next_gen = algorithms.varOr(population =self.pop['connWeights_HH_pop'],
                                                                                            toolbox = self.varTbox['real'],
                                                                                            lambda_ = self.params['conn_pop_size'],
                                                                                            cxpb = 1-self.params['prob_mutation'],
                                                                                             mutpb = self.params['prob_mutation']
                                        );
                                elif self.params['variation'] == 'varAnd':
                                        next_gen = algorithms.varAnd(population = self.pop['connWeights_HH_pop'],
                                                                                          toolbox = self.varTbox['real'],
                                                                                          cxpb  = self.params['prob_crossOver'],
                                                                                          mutpb = self.params['prob_mutation']);
                                #evaluate
                                self.evaluate_weightHH(next_gen);

                                #next generation
                                if next_gen == None:
                                    next_gen = self.pop['connWeights_HH_pop'];

                                #selection of next population
                                self.pop['connWeights_HH_pop'] = tools.selTournament(next_gen,
                                                                             self.params['conn_pop_size'],
                                                                             self.params['tourn_size']);


                        #selection of next population
                        next_repr['connWeights_HH'] = self.tbox.clone(tools.selBest(self.pop['connWeights_HH_pop'],1)[0]);

                ##################CONN_WEIGHTS HIDDEN TO OUTPUT LAYER ######################
                if s_key == 'connWeights_HO_pop':
                        #evaluate members of population
                        for weights in specie:
                            #clone repr
                            sol = self.tbox.clone(self.represntatives);
                            #replace
                            sol['connWeights_HO'] = weights;
                            #evaluate
                            f = self.evaluate(sol,copy = False);
                            weights.fitness.values = f[0],f[1];
                            if debug:
                                print "connWeightsHO fitness:", weights.fitness.values[0];
                            del sol;



                        #Should we optimise?
                        next_gen = None;
                        if not self.params['optimFreeze_weightsHO']:
                                print "-optim weightsHO";

                                #Variations (Cross over and Mutation)
                                if self.params['variation'] =='varOr': #select (parents + children)
                                        try:
                                            next_gen = algorithms.varOr(population =self.pop['connWeights_HO_pop'],
                                                                                                toolbox = self.varTbox['real'],
                                                                                                lambda_ = self.params['conn_pop_size'],
                                                                                                cxpb = 1-self.params['prob_mutation'],
                                                                                                 mutpb = self.params['prob_mutation']
                                            );
                                        except ValueError:
                                            ## Mutation Only ##
                                            #get candidates and clone
                                            candidates = [self.tbox.clone(w_ho) for w_ho in self.pop['connWeights_HO_pop']];
                                            #mutate each one of them according to prob of params set.
                                            mutants = [self.varTbox['real'].mutate(c_i) for c_i in candidates];
                                            #items are returned as tuple from mutate operation so we unpack them
                                            next_gen = [m_i[0] for m_i in mutants];

                                elif self.params['variation'] == 'varAnd':
                                        try:
                                            next_gen = algorithms.varAnd(population = self.pop['connWeights_HO_pop'],
                                                                                          toolbox = self.varTbox['real'],
                                                                                          cxpb  = self.params['prob_crossOver'],
                                                                                          mutpb = self.params['prob_mutation']);
                                        except ValueError:
                                            ## Mutation Only ##
                                            #get candidates and clone
                                            candidates = [self.tbox.clone(w_ho) for w_ho in self.pop['connWeights_HO_pop']];
                                            #mutate each one of them according to prob of params set.
                                            mutants = [self.varTbox['real'].mutate(c_i) for c_i in candidates];
                                            #items are returned as tuple from mutate operation so we unpack them
                                            next_gen = [m_i[0] for m_i in mutants];

                                #evaluate
                                self.evaluate_weightHO(next_gen);


                                #next generation
                                if next_gen == None:
                                    next_gen = self.pop['connWeights_HO_pop'];

                                #selection of next population
                                self.pop['connWeights_HO_pop'] = tools.selTournament(next_gen,
                                                                             self.params['conn_pop_size'],
                                                                             self.params['tourn_size']);

                        #selection of next population
                        next_repr['connWeights_HO'] = self.tbox.clone(tools.selBest(self.pop['connWeights_HO_pop'],1)[0]);


            # Copy representation
            self.represntatives = self.tbox.clone(next_repr);
            #store best cost
            self.best_cost_lst.append(self.best_cost);

            ################## PLOT ######################
            # if self.current_iter % self.plot_interval == 0:
            #
            #     #Plot error
            #     plt.plot(self.best_cost_lst,'--b',label = 'best cost (model)');
            #     plt.draw();

                #todo: take snapshot of generations (statistics)
                #todo: visualisation of current best network (with snapshots) - On Demand


        ################## *** OPTIMISATION COMPLETE *** ######################
        print "Optimisation Complete!!!";
        print "id(sol)", id(self.best_sol);
        print "sol:", self.best_sol;
        print "best_cost(train):", self.best_cost;
        #Test
        train_err = self.best_model.evaluate(self.train_set);
        self.best_model.flush();
        test_err = self.best_model.evaluate(self.test_set);
        self.best_cost_test = test_err;
        print "best_cost(train):",train_err;
        print "best_cost(test):",test_err;
        self.best_model.flush();

        ##STORE STATISTICS ###
        #- get statistics
        #gather some statistics
            # - hidden nodes
        # for i in self.pop['hidden_nodes'].keys():
        #     self.hl_stats.compile(self.pop['hidden_nodes'][i]);
        #     #-output nodes
        # for i in self.pop['out_nodes'].keys():
        #     self.hl_stats.compile(self.pop['out_nodes'][i]);
        #     #- model
        # self.model_stats.compile(self.pop['model_pop']);
        #     #-topology
        # self.conn_stats.compile(self.pop['connActive_IH_pop']);
        # self.conn_stats.compile(self.pop['connActive_HH_pop']);
        # self.conn_stats.compile(self.pop['connActive_HO_pop']);
        #     #-weights
        # self.weights_stats.compile(self.pop['connWeights_IH_pop']);
        # self.weights_stats.compile(self.pop['connWeights_HH_pop']);
        # self.weights_stats.compile(self.pop['connWeights_HO_pop']);

        # - store records
        # - get time stamp
        timestamp = time.ctime();
        timestamp = timestamp.replace(' ','_');
        timestamp = timestamp.replace(':','');


        #Form pre string for filenames
        preString = self.dataset_name + '_' +  timestamp+ '_';
        # - Store statistics
        DataFrame( data = self.best_cost_lst, columns = ['best_cost']).to_csv(preString+'gen_fitness.csv');
        for k, v in self.snapshot.iteritems():
            DataFrame(data = self.snapshot[k], columns = [k]).to_csv(preString+ k + '.csv');

        #-store best models performance
        performance = {'train_err': [train_err],
                       'test_err': [test_err]};
        DataFrame(performance).to_csv(preString+'Performance.csv');

        # store settings
        exp_params = dict();
        exp_params['NGEN'] = self.params['NGEN'];
        exp_params['numI'] = self.params['numI'];
        exp_params['numH'] = self.params['numH'];
        exp_params['numO'] = self.params['numO'];
        exp_params['optimFreeze_weightsHH'] = self.params['optimFreeze_weightsHH'];
        exp_params['optimFreeze_weightsHO'] = self.params['optimFreeze_weightsHO'];
        exp_params['optimFreeze_weightsIH'] = self.params['optimFreeze_weightsIH'];
        exp_params['optimFreeze_connIH'] = self.params['optimFreeze_connIH'];
        exp_params['optimFreeze_connHH'] = self.params['optimFreeze_connHH'];
        exp_params['optimFreeze_connHO'] = self.params['optimFreeze_connHO'];
        exp_params['model_pop_size'] = self.params['model_pop_size'];
        exp_params['node_pop_size'] = self.params['node_pop_size'];
        exp_params['conn_pop_size'] = self.params['conn_pop_size'];
        exp_params['initConnectivity'] = self.params['initConnectivity'];
        exp_params['initTransferFn'] = self.params['initTransferFn'];
        exp_params['prodConstantMax'] = self.params['prodConstantMax'];
        #SAVE EXP Params
        expParamsFile = open(preString+'EXP_params.dat','w');
        pickle.dump(exp_params,expParamsFile );
        expParamsFile.close();

        ####STORE BEST MODEL###
        fstore = open(self.dataset_name+'best_model'+timestamp+'.dat', 'w');
        pickle.dump(
            {'best_sol': self.best_sol,
             'best_cost': self.best_cost},fstore);

        fstore.close();
        #plot model performance
        # vis2d.visualiseOutputs2D(self.best_model,self.train_set);
        ### VISUALISE THE NETWORK ###
        # app = QtGui.QApplication(sys.argv);
        # QtCore.qsrand(QtCore.QTime(0,0,0).secsTo(QtCore.QTime.currentTime()));
        # widget2 = GraphWidget(self.best_model,'Train Set');
        # widget2.show();
        # sys.exit(app.exec_());



        ################## RETURN ######################
        return self.best_model;



