__author__ = 'Abdullah'

import sys;


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
from benchmark import lab_bencmark;
import commons;
import time;
import cPickle as pickle;
#import visualisation.visualiseOutputs2D as vis2d;
from deap import tools, algorithms;
from scoop import futures;



#import visualiseNet
try:
    from visualiseNDMNet import *;
    visualiseNet = False;
except ImportError:
    visualiseNet = True;


"""
Initialise
"""

#statisitics
#-fitness
stats_fitness = tools.Statistics(key = lambda sol: sol.fitness.values);
#-connectivity statistics
conn_stats = tools.MultiStatistics(fitness = stats_fitness);
#-weights statistics
weights_stats = tools.MultiStatistics(fitness = stats_fitness);
#-hidden nodes statistics
hl_stats = tools.MultiStatistics(fitness = stats_fitness);
#-output node statistics
ol_stats = tools.MultiStatistics(fitness = stats_fitness);
#-model
stats_lateral_conn = tools.Statistics(key = lambda model: model[0]); # Lateral connections
stats_context_layer = tools.Statistics(key = lambda model: model[1]); # Context layer
stats_gauss_noise = tools.Statistics(key = lambda model: model[2]); # Gaussian Noise

model_stats = tools.MultiStatistics(lateral_conn = stats_lateral_conn,
                                    context_layer = stats_context_layer,
                                    gauss_noise = stats_gauss_noise,
                                    fitness = stats_fitness);

####
log = tools.Logbook();



comp_best_cost = {
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
lab_data = lab_bencmark();

print ">>>IRIS";

D = lab_data.iris();

dataset_name = 'IRIS';
train_set = D['train'];
test_set = D['test'];
validation_set = D['valdidation'];




#POPULATION
pop = {
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
represntatives = None;


#Optimisation params - BIAS
params = dict();
params['targer_err'] = 0.00;
params['NGEN'] = 100;
### NET CONFIG
params['numI'] = 2; #number of input nodes
params['numH'] = 4; #number of hidden nodes
params['numO'] = 1; #number of output nodes

params['randomNodesInject'] = True; #DEFAULT
params['temporalNodesInject'] = False; #temporal nodes injection - Switch ON before optim

params['select_mut'] = 1;
params['select_nextGen'] = 1;
params['select_crossOver'] = 1;
params['select_diffEvo'] = 1;

params['initConnectivity'] = 'full'; #connectivity {'random', 'full'}

params['noIntraLayerConnHH'] = False; #no lateral connections between hidden units #TODO : update this to repo (3/7/14) - [Not Done ]

params['noContextLayer'] = False; #no context layer for the hidden units #TODO : update this to repo (3/7/14) - [Not Done ]

params['noNoise'] = False; # no gaussian noise for the hidden units #TODO : update this to repo (3/7/14) - [Not Done ]
params['initTransferFn'] = 'all'; #random: randomly creates transfer functions, and doesn't account for duplication, #all: creates all possible combinations of transfer functions
params['output_fns'] = [   outFn.identity_fn,
                           outFn.sigmoid_fn,
                           outFn.gaussian_fn,
                           outFn.tanh_fn,
                           outFn.arc_tanh_fn,
                           outFn.gaussian_ii_fn

]; #Global output functions

params['activation_fns'] = [inFn.max_dist,
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

params['hl_output_fns'] = [outFn.identity_fn,
                               outFn.sigmoid_fn,
                               outFn.gaussian_fn,
                               outFn.tanh_fn,
                               outFn.arc_tanh_fn,
                               outFn.gaussian_ii_fn
    ]; #hidden layer output functions


params['ol_output_fns'] = [outFn.identity_fn,
                               outFn.sigmoid_fn,
                               outFn.gaussian_fn,
                               outFn.tanh_fn,
                               outFn.arc_tanh_fn,
                               outFn.gaussian_ii_fn
    ]; #output layer output functions


params['hl_activation_fns'] = [
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

params['ol_activation_fns'] = [
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
params['inject_sequence'] = ['projection', 'radial-basis', 'higher-order'];
params['inject_trigger'] = 'interval'; # options 'stagnation', 'interval'
params['temporal_change_interval'] = 25;
params['curr_inject_sequence'] = 0;
#- injection set
params['hl_inject_act_fns'] = [];
params['hl_inject_out_fns'] = [];
params['ol_inject_act_fns'] = [];
params['ol_inject_out_fns'] = [];

### Population size
params['node_pop_size'] = 30;
params['conn_pop_size'] = 30;
params['model_pop_size'] = 30;
params['prodConstantMax'] = 5;
params['greedThreshold'] = 3; #For funnel search
params['variation'] = 'varAnd'; #options : {varOr or varAnd} varOr - sample from (parent + child), varAnd samples from (child)
params['percentNextGen'] = 0.5; #percentage of the population to select for next generation
params['DEIterMax'] = 4; #max differential evolution iterations
params['target_error'] = 0.0; # targer error
params['prob_crossOver'] = 0.4; #probability of crossover
params['prob_co_indp'] = 0.5; #probability of crossing over individual parameter
params['prob_mutation'] = 0.6; #probablity of mutation
params['prob_mut_indp'] = 0.2; #probability of mutating individual parameters
params['alpha'] = 0.2; #significant of velocities for differential evolution
params['param_min'] = -1.0; #parameter min
params['param_max'] = 1.0; #parameter max
params['do_diffEvolution'] = True; #does differential evolution
params['do_crossOver'] = True; #cross over
params['do_mutation'] = True; #mutation
#Freeze component optimisation
#-model
params['optimFreeze_model'] = False;
#-nodes
params['optimFreeze_hidNodes'] = False;
params['optimFreeze_outNodes'] = False;
#-connectivity
params['optimFreeze_connIH'] = True;
params['optimFreeze_connHH'] = False;
params['optimFreeze_connHO'] = True;
#-weights
params['optimFreeze_weightsIH'] = False;
params['optimFreeze_weightsHH'] = False;
params['optimFreeze_weightsHO'] = False;
#mutation
params['gaus_mut_mean'] = 0.1; #mutaion mean
params['gaus_mut_std'] = 0.2; #mutation std. deviation
#evolutionary algorithm
params['tourn_size'] = 3;
params['ea_ngen'] = 3;
params['ea_mu_model'] = int(0.5 * params['model_pop_size']); #number of solutions to select for next generation
params['ea_lamda_model'] = params['model_pop_size'] - params['ea_mu_model']; #pop to produce for next gen.
params['ea_mu_nodes'] = int(0.5 * params['node_pop_size']); #number of solutions to select for next generation
params['ea_lamda_nodes'] = params['node_pop_size'] - params['ea_mu_nodes']; #pop to produce for next gen.
params['ea_mu_conn'] = int(0.5 * params['conn_pop_size']); #number of solutions to select for next generation
params['ea_lamda_conn'] = params['conn_pop_size'] - params['ea_mu_conn']; #pop to produce for next gen.

#TOOLBOX
tbox = ndmCoevoBase.init_toolbox(params);
varTbox = coevoOptim.getToolbox(params);

#NO TRAINING
noTraining = False;
if noTraining:
    params['optimFreeze_model'] = True;
    #-nodes
    params['optimFreeze_hidNodes'] = True;
    params['optimFreeze_outNodes'] = True;
    #-connectivity
    params['optimFreeze_connIH'] = True;
    params['optimFreeze_connHH'] = True;
    params['optimFreeze_connHO'] = True;
    #-weights
    params['optimFreeze_weightsIH'] = True;
    params['optimFreeze_weightsHH'] = True;
    params['optimFreeze_weightsHO'] = True;

def gen_random_nodes(self,layer, n = 5):
    """
    generate random nodes with random transfer functions
    """

    rnd_nodes = [tbox.node() for _ in xrange(n)];
    if layer == 'hidden':
        ndmCoevoBase.init_tf_fn_forPop(rnd_nodes , params =params, layer = 'hidden');
    elif layer == 'output':
        ndmCoevoBase.init_tf_fn_forPop(rnd_nodes, params = params, layer = 'output');
    else:
        raise Exception("Invalid layer specification in gen_random_nodes()");




    return rnd_nodes;

def gen_temporal_nodes(self,layer, n = 5):
    """
    generate random nodes from the current temporal pattern
    """

    #create the new nodes
    tmp_nodes = [tbox.node() for _ in xrange(n)];

    #get the current pattern for the current tempo.
    fn_class = None;
    if current_iter % params['temporal_change_interval'] == 0:


        #increament and change
        if params['curr_inject_sequence'] < len(params['inject_sequence'])-1:
            #change then increament
            params['curr_inject_sequence'] +=1
        else:
            #start from begining
            params['curr_inject_sequence'] = 0;


        #change
        fn_class = params['inject_sequence'][params['curr_inject_sequence']];

        #verbose
        print ">>fn_class", fn_class;

        #switch transfer function set used for optimisation
        params['hl_activation_fns'] = inFn.getFunctionsClass(fn_class);
        params['hl_output_fns_fns'] = outFn.getFunctionsClass(fn_class);
        params['ol_activation_fns'] = inFn.getFunctionsClass(fn_class);
        params['ol_output_fns'] = outFn.getFunctionsClass(fn_class);

    #set their transfer functions
    if layer == 'hidden':
        ndmCoevoBase.init_tf_fn_forPop(tmp_nodes , params = params, layer = 'hidden');
    elif layer == 'output':
        ndmCoevoBase.init_tf_fn_forPop(tmp_nodes,  params =params, layer = 'output');
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
    sol['connActive_IH'] = commons.matrixForm(components['connActive_IH'],params['numI'], params['numH']);
    sol['connActive_HH'] = commons.matrixForm(components['connActive_HH'],params['numH'], params['numH']);
    sol['connActive_HO'] = commons.matrixForm(components['connActive_HO'],params['numH'], params['numO']);
    #weights
    sol['connWeights_IH'] = commons.matrixForm(components['connWeights_IH'],params['numI'], params['numH']);
    sol['connWeights_HH'] = commons.matrixForm(components['connWeights_HH'],params['numH'], params['numH']);
    sol['connWeights_HO'] = commons.matrixForm(components['connWeights_HO'],params['numH'], params['numO']);
    #unpack nodes
    sol['hidden_nodes'] = [node for nkey,node in components['hidden_nodes'].iteritems() ];
    sol['out_nodes'] = [node for nkey,node in components['out_nodes'].iteritems() ];

    try:
        #create model
        model = ndmModel(numI = params['numI'],
                         numH =  params['numH'],
                         numO = params['numO'],
                         components = sol,
                         noIntraLayerConnHH = params['noIntraLayerConnHH'],
                         noContextLayer = params['noContextLayer'],
                         noNoise = params['noNoise']); #TODO : update this to repo (3/7/14) - [Not Done ]

    except AttributeError:
        raise Exception("Oopps - something went wrong while evaluating");

    return model;


def evaluate(self, components, verbose = False, copy = True, eval_type = None,**kwargs):
    """
    Evaluate a given solution of components from all species
    """

    #fitnesses
    fitness_test = 1;
    fitness_val = 1;


    if copy:
        sol_eval = tbox.clone(components);
    else:
        sol_eval = components;

    try:

        #Make model
        model_eval = create_model(sol_eval);
        #evaluate on training set
        fitness_train  = model_eval.evaluate(train_set);

        if eval_type != None:

            ### EVALUATE ON TEST SET
            if eval_type == 'test':
                fitness_test  = model_eval.evaluate(test_set);


            ### EVALUATE ON VALIDATION SET
            if eval_type == 'validation':
                fitness_val = model_eval.evaluate(validation_set);


    except AttributeError:
        raise Exception("Oopps - something went wrong while evaluating");

    #stor best cost
    if fitness_train < best_cost:
        #store best cost
        best_cost = fitness_train;
        print "Best_cost:", best_cost;

        #store solutions every couple of generations
        best_sol = (sol_eval);
        best_model = create_model(sol_eval);

    if fitness_val < best_cost_val:
        #store the best cots (validation)
        best_cost_val = fitness_val;
        print "Best_cost(validation)", best_cost_val;

        best_sol_val = sol_eval;
        best_model_val = create_model(sol_eval);

    if fitness_test < best_cost_test:
        #store the best cost (test)
        best_cost_test = fitness_test;
        print "Best_cost(test)", best_cost_test;
        best_sol_test = sol_eval;
        best_model_test = create_model(sol_eval);


    return fitness_train,fitness_val,fitness_test;


def evaluate_map(self, components, verbose = False, copy = True, eval_type = None,**kwargs):
    """
    Evaluate a given solution of components from all species
    """

    #fitnesses
    fitness_test = 1;
    fitness_val = 1;


    if copy:
        sol_eval = tbox.clone(components);
    else:
        sol_eval = components;

    try:

        #Make model
        model_eval = create_model(sol_eval);
        #evaluate on training set
        fitness_train  = model_eval.evaluate(train_set);

        if eval_type != None:

            ### EVALUATE ON TEST SET
            if eval_type == 'test':
                fitness_test  = model_eval.evaluate(test_set);


            ### EVALUATE ON VALIDATION SET
            if eval_type == 'validation':
                fitness_val = model_eval.evaluate(validation_set);


    except AttributeError:
        raise Exception("Oopps - something went wrong while evaluating");

    print "Fitness:", fitness_test, fitness_train, fitness_val;
    # #stor best cost
    # if fitness_train < kwargs['best_cost']:
    #     #store best cost
    #     best_cost = fitness_train;
    #     print "Best_cost:", best_cost;
    #
    #     #store solutions every couple of generations
    #     best_sol = (sol_eval);
    #     best_model = create_model(sol_eval);
    #
    # if fitness_val < kwargs['best_cost_val']:
    #     #store the best cots (validation)
    #     best_cost_val = fitness_val;
    #     print "Best_cost(validation)", best_cost_val;
    #
    #     best_sol_val = sol_eval;
    #     best_model_val = create_model(sol_eval);
    #
    # if fitness_test < kwargs['best_cost_test']:
    #     #store the best cost (test)
    #     best_cost_test = fitness_test;
    #     print "Best_cost(test)", best_cost_test;
    #     best_sol_test = sol_eval;
    #     best_model_test = create_model(sol_eval);


    return fitness_train,fitness_val,fitness_test;

def evaluate_connIH_map(self,represntatives,connIH):
    """
    evaluates an input-hidden connection topology in parallel using dtm - (MPI)
    """
    sol = tbox.clone(represntatives);

    if connIH.fitness.valid != True:
        sol['connActive_IH'] = connIH;
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        connIH.fitness.values = f[0],f[1];

    del sol;

    return connIH;

def evaluate_connIH(self, next_gen):
    """
    This method unlike the evaluate method evaluates a single component
    """

    sol = tbox.clone(represntatives);

    for connIH in next_gen:
        if connIH.fitness.valid != True:
            sol['connActive_IH'] = connIH;
            f = evaluate(sol,copy = False, eval_type= 'validation');
            connIH.fitness.values = f[0],f[1];


            #Save best cost
            last = len(snapshot['topology']) -1;
            if f[0] < snapshot['topology'][last]:
                snapshot['topology'].append(f[0]);

    del sol;


def evaluate_connHH_map(self,represntatives,connHH):
    """
    evaluate hidden to hidden connections in parallel
    """

    sol = tbox.clone(represntatives);

    if connHH.fitness.valid != True:
        sol['connActive_HH'] = connHH;
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        connHH.fitness.values = f[0],f[1];


    del sol;

    return connHH;

def evaluate_connHH(self, next_gen):
    """
    evaluate hidden to hidden connections
    """

    sol = tbox.clone(represntatives);

    for connHH in next_gen:
        if connHH.fitness.valid != True:
            sol['connActive_HH'] = connHH;
            f = evaluate(sol,copy = False, eval_type= 'validation');
            connHH.fitness.values = f[0],f[1];


            #Save best cost
            last = len(snapshot['topology']) -1;
            if f[0] < snapshot['topology'][last]:
                snapshot['topology'].append(f[0]);


    del sol;


def evaluate_connHO_map(self,represntatives,connHO):
    """
     parallel evaluation of the hidden to output units
    """
    sol = tbox.clone(represntatives);

    if connHO.fitness.valid != True:
        sol['connActive_HO'] = connHO;
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        connHO.fitness.values = f[0],f[1];


    del sol;

    return connHO;

def evaluate_connHO(self, connHO):
    """
    connHO
    """
    sol = tbox.clone(represntatives);

    if connHO.fitness.valid != True:
        sol['connActive_HO'] = connHO;
        f = evaluate(sol,copy = False, eval_type= 'validation');
        connHO.fitness.values = f[0],f[1];


    del sol;


def evaluate_weightIH_map(self,represntatives,weightIH):
    """
    evaluates in parallel
    """
    sol = tbox.clone(represntatives);

    if weightIH.fitness.valid != True:
        sol['connWeights_IH'] = weightIH;
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        weightIH.fitness.values = f[0],[1];


    #free up memory
    del sol;

    return weightIH;


def evaluate_weightIH(self,next_gen):
    """
    evaluates a given set of weights
    """
    #make a copy
    sol = tbox.clone(represntatives);

    for weightIH in next_gen:
        if weightIH.fitness.valid != True:
            sol['connWeights_IH'] = weightIH;
            f = evaluate(sol,copy = False, eval_type= 'validation');
            weightIH.fitness.values = f[0],[1];

            #Save best cost
            last = len(snapshot['weights']) -1;
            if f[0] < snapshot['weights'][last]:
                snapshot['weights'].append(f[0]);

    #free up memory
    del sol;

def evaluate_weightHH_map(self, represntatives, weightHH):
    """
    """

    sol = tbox.clone(represntatives);

    if weightHH.fitness.valid != True:
        sol['connWeights_HH'] = weightHH;
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        weightHH.fitness.values = f[0],f[1];

    #free up memory
    del sol;

    return weightHH;

def evaluate_weightHH(self,next_gen):
    """
    evaluates the lateral connections
    """

    #make a copy
    sol = tbox.clone(represntatives);

    for weightHH in next_gen:
        if weightHH.fitness.valid != True:
            sol['connWeights_HH'] = weightHH;
            f = evaluate(sol,copy = False, eval_type= 'validation');
            weightHH.fitness.values = f[0],f[1];

            #Save best cost
            last = len(snapshot['weights']) -1;
            if f[0] < snapshot['weights'][last]:
                snapshot['weights'].append(f[0]);

    #free up memory
    del sol;



def evaluate_weightHO_map(self, weightsHO):
    """
    evaluates weights from hidden to output
    """

    sol = tbox.clone(represntatives);


    if weightsHO.fitness.valid != True:
        #replace
        sol['connWeights_HO'] = weightsHO;
        #evaluate
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        weightsHO.fitness.values = f[0],f[1];


    del sol;

    return  weightsHO;


def evaluate_weightHO(self, next_gen):
    """
    evaluates weights from hidden to output
    """

    sol = tbox.clone(represntatives);

    for weightsHO in next_gen:
        if weightsHO.fitness.valid != True:
            #replace
            sol['connWeights_HO'] = weightsHO;
            #evaluate
            f = evaluate(sol,copy = False, eval_type= 'validation');
            weightsHO.fitness.values = f[0],f[1];


            #Save best cost
            last = len(snapshot['weights']) -1;
            if f[0] < snapshot['weights'][last]:
                snapshot['weights'].append(f[0]);

    del sol;

def evaluate_model_map(self,represntatives, model):
    """
    Evaluate models in parallel
    """
    sol = tbox.clone(represntatives);

    if model.fitness.valid != True:
        sol['model'] = model;
        #evaluate the model
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        model.fitness.values = f[0],f[1];

    #GC
    del sol;

    return model;




def evaluate_model(self,next_gen):
    """
    evaluates a given model
    """

    #make a copy
    sol = tbox.clone(represntatives);

    #evaluate all models
    for model in next_gen:
        if model.fitness.valid != True:
            sol['model'] = model;
            #evaluate the model
            f = evaluate(sol,copy = False, eval_type= 'validation');
            model.fitness.values = f[0],f[1];

            #Save best cost
            last = len(snapshot['model']) -1;
            if f[0] < snapshot['model'][last]:
                snapshot['model'].append(f[0]);

    #GC
    del sol;


def evaluate_hiddenNodes_map(self,represntatives,h_node, hi):
    """
    implementation that works using mpi for parallel computing
    """

    sol = tbox.clone(represntatives);

    if h_node.fitness.valid != True:
        #replace
        sol['hidden_nodes'][hi] = h_node;
        #evaluate
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        h_node.fitness.values = f[0],f[1]


    #GC
    del sol;
    return h_node;

def evaluate_hiddenNodes(self,next_gen,hi):
    """
    evaluates a hidden node
    """
    sol = tbox.clone(represntatives);

    for h_node in next_gen:

        if h_node.fitness.valid != True:
            #replace
            sol['hidden_nodes'][hi] = h_node;
            #evaluate
            f = evaluate(sol,copy = False, eval_type= 'validation');
            h_node.fitness.values = f[0],f[1];

            #Save best cost
            last = len(snapshot['hidden_layer']) -1;
            if f[0] < snapshot['hidden_layer'][last]:
                snapshot['hidden_layer'].append(f[0]);

    #GC
    del sol;


def evaluate_outputNodes_map(self, represntatives,o_node, oi):
    """
    """

    sol = tbox.clone(represntatives);

    if o_node.fitness.valid != True:
        #replace
        sol['out_nodes'][oi] = o_node;
        #evaluate
        f = evaluate_map(sol,copy = False, eval_type= 'validation');
        o_node.fitness.values = f[0],f[1];


    #GC
    del sol;

    return o_node;


def evaluate_outputNodes(self, next_gen, oi):
    """
    evaluate output nodes
    """

    sol = tbox.clone(represntatives);

    for o_node in next_gen:

        if o_node.fitness.valid != True:
            #replace
            sol['out_nodes'][oi] = o_node;
            #evaluate
            f = evaluate(sol,copy = False, eval_type= 'validation');
            o_node.fitness.values = f[0],f[1];

            #Save best cost
            last = len(snapshot['output_layer']) -1;
            if f[0] < snapshot['output_layer'][last]:
                snapshot['output_layer'].append(f[0]);


    #GC
    del sol;


def init_populations(self):
    """
    Intialises the populations based on the parameter settings
    """

    #hidden nodes
    for ni in xrange(params['numH']):
        if params['initTransferFn'] == 'random':
            pop['hidden_nodes'][str(ni)] = [tbox.node() for _ in xrange(params['node_pop_size'])];
        elif params['initTransferFn'] == 'all':
            tfs_comb =len([(w,n) for w in params['hl_activation_fns'] for n in params['hl_output_fns']]);
            pop['hidden_nodes'][str(ni)] = [tbox.node() for _ in xrange(tfs_comb)];

    #output nodes
    for ni in xrange(params['numO']):

        if params['initTransferFn'] == 'random':
            pop['out_nodes'][str(ni)] = [tbox.node() for _ in xrange(params['node_pop_size'])];
        elif params['initTransferFn'] == 'all':
            tfs_comb =len([(w,n) for w in params['ol_activation_fns'] for n in params['ol_output_fns']]);
            pop['out_nodes'][str(ni)] = [tbox.node() for _ in xrange(tfs_comb)];

    #initialise transfer functions
    if params['initTransferFn'] == 'all':
        ndmCoevoBase.init_all_transferFns(pop['hidden_nodes'], params = params, layer = 'hidden');
        ndmCoevoBase.init_all_transferFns(pop['out_nodes'], params = params, layer = 'output');
    elif params['initTransferFn'] == 'random':
        ndmCoevoBase.init_tf_fn_random(pop['hidden_nodes'], params =params, layer = 'hidden');
        ndmCoevoBase.init_tf_fn_random(pop['out_nodes'], params = params, layer = 'output');
    else:
        raise Exception("No such initTransferFn method, use 'all' or 'random'");


    #connectivity/toplogy
    pop['connActive_IH_pop'] = [tbox.connActive_IH() for _ in xrange(params['conn_pop_size'])];
    pop['connActive_HH_pop'] = [tbox.connActive_HH() for _ in xrange(params['conn_pop_size'])];
    pop['connActive_HO_pop'] = [tbox.connActive_HO() for _ in xrange(params['conn_pop_size'])];
    #weights
    pop['connWeights_IH_pop'] = [tbox.connWeights_IH() for _ in xrange(params['conn_pop_size'])];
    pop['connWeights_HH_pop'] = [tbox.connWeights_HH() for _ in xrange(params['conn_pop_size'])];
    pop['connWeights_HO_pop'] = [tbox.connWeights_HO() for _ in xrange(params['conn_pop_size'])];
    #models
    pop['model_pop'] = [tbox.model() for _ in xrange(params['model_pop_size'])];


def init_empty_population(self):
    """
    Initialise and empty population
    """

    #hidden nodes
    for ni in xrange(params['numH']):
        pop['hidden_nodes'][str(ni)] = [];

    #output nodes
    for ni in xrange(params['numO']):
        pop['out_nodes'][str(ni)] = [];


    #connectivity/toplogy
    pop['connActive_IH_pop'] = [];
    pop['connActive_HH_pop'] = [];
    pop['connActive_HO_pop'] = [];
    #weights
    pop['connWeights_IH_pop'] = [];
    pop['connWeights_HH_pop'] = [];
    pop['connWeights_HO_pop'] = [];
    #models
    pop['model_pop'] = [];




def select_representative(self):
    """
    Returns a dictionary of a random set of representatives from all the species
    """
    #Select representatives randomly
    representative = dict();
    representative['model'] = tools.selRandom(pop['model_pop'],1)[0];
    representative['connActive_IH'] = tools.selRandom(pop['connActive_IH_pop'],1)[0];
    representative['connActive_HH'] = tools.selRandom(pop['connActive_HH_pop'],1)[0];
    representative['connActive_HO'] = tools.selRandom(pop['connActive_HO_pop'],1)[0];
    representative['connWeights_IH'] = tools.selRandom(pop['connWeights_IH_pop'],1)[0];
    representative['connWeights_HH'] = tools.selRandom(pop['connWeights_HH_pop'],1)[0];
    representative['connWeights_HO'] = tools.selRandom(pop['connWeights_HO_pop'],1)[0];

    representative['hidden_nodes'] = dict();
    for k,v in pop['hidden_nodes'].iteritems():
        representative['hidden_nodes'][str(k)] = tools.selRandom(v,1)[0];

    representative['out_nodes'] = dict();
    for k,v in pop['out_nodes'].iteritems():
        representative['out_nodes'][str(k)] = tools.selRandom(v,1)[0];


    return representative;



#### MAIN ####
if __name__ == '__main__':
    """
    Optimise neural network architectures
    """
    ##
    best_cost = float('Inf');
    best_cost_val = float('Inf');
    best_cost_test = float('Inf');

    #params
    snapshot_interval = 1;
    snapshot = {
        'hidden_layer': [float('inf')],
        'output_layer': [float('inf')],
        'topology': [float('inf')],
        'weights': [float('inf')],
        'model': [float('inf')]
    };
    current_iter = 0;
    best_cost = float('inf');
    best_cost_val = float('inf');
    best_cost_test = float('inf');

    best_cost_lst = [];
    best_cost_val_lst = [];

    best_sol = None;
    best_sol_val = None;
    best_sol_test = None;
    best_model = None;
    plot_interval = 2;


    ### Optimisation ###

    #select representative
    represntatives = select_representative();
    #next representative should clone the current representative components
    next_repr = tbox.clone(represntatives);


    ## PRINT OUT SOME OPTIMISATION PARAMETERS - FOR CONFIRMATION #TODO : update this to repo (3/7/14) - [Not Done ]
    print " ################ Optimisation Params ################"
    for param, value in params.iteritems():
        print "> param: ", param, " : ", value;

    #Coevolution Loop
    for current_iter in xrange(params['NGEN']):


        #Check stopping criteria
        # if round(best_cost,2) < params['targer_err']:
        #     #STOP THE OPTIMISATION
        #     break;

        print "GEN", current_iter;


        #Iterate over species
        for s_key, specie in pop.iteritems():

            #Use representatives to evaluate the individual in the current specie
            if s_key == 'hidden_nodes':

                #for all the hidden nodes positions
                for hi,h_nodes in specie.iteritems():
                    #evaluate candidates for that pos.
                    h_nodes = futures.map(evaluate_hiddenNodes_map, represntatives, h_nodes,hi);

                    #should optimise?
                    next_gen = None;
                    if not params['optimFreeze_hidNodes']:
                        print "-optim hidNodes";
                        #Variations (Cross over and Mutation)
                        if params['variation'] == 'varOr': #variate and select from  (parent + offsprings)
                            next_gen = algorithms.varOr(population = pop['hidden_nodes'][hi],
                                                        toolbox = varTbox['real'],
                                                        lambda_  = params['node_pop_size'],
                                                        cxpb = 1 - params['prob_mutation'],
                                                        mutpb = params['prob_mutation']);
                        elif params['variation'] == 'varAnd':
                            next_gen = algorithms.varAnd(population = pop['hidden_nodes'][hi],
                                                         toolbox = varTbox['real'],
                                                         cxpb = params['prob_crossOver'],
                                                         mutpb = params['prob_mutation']
                            );

                        #generate random nodes
                        if params['randomNodesInject']:
                            print ">>inject";
                            rand_nodes = gen_random_nodes(layer = 'hidden',n = 10);
                            #add to next gen
                            next_gen.extend(rand_nodes);

                        #generate temporal nodes
                        if params['temporalNodesInject']:
                            print ">>Inject - Temporal patterning";
                            temp_nodes = gen_temporal_nodes(layer = 'hidden', n = 10);
                            #add to next gen
                            next_gen.extend(temp_nodes);


                        #Evaluate
                        evaluate_hiddenNodes(next_gen,hi);

                        #next generation
                        if next_gen == None:
                            next_gen = pop['hidden_nodes'][hi];

                        #selection of next population
                        pop['hidden_nodes'][hi] = tools.selTournament(next_gen,
                                                                      params['node_pop_size'],
                                                                      params['tourn_size']);

                    #selection of next representative
                    best_hidden_node = tools.selBest(pop['hidden_nodes'][hi],1)[0];
                    next_repr['hidden_nodes'][hi] = tbox.clone(best_hidden_node);



            ##################OUTPUT NODES ######################
            if s_key == 'out_nodes':
                #for all the output nodes
                for oi,o_nodes in specie.iteritems():
                    #evaluate candidate members for that output node position
                    o_nodes = futures.map(evaluate_outputNodes_map, represntatives,o_nodes, oi );

                    #should optimise?
                    next_gen = None;
                    if not params['optimFreeze_outNodes']:
                        print "-optim outNodes";
                        #Variations (Cross over and Mutation)
                        if params['variation'] == 'varOr': #variate and select from  (parent + offsprings)
                            next_gen = algorithms.varOr(population = pop['out_nodes'][oi],
                                                        toolbox = varTbox['real'],
                                                        lambda_  = params['node_pop_size'],
                                                        cxpb = 1 - params['prob_mutation'],
                                                        mutpb = params['prob_mutation']);
                        elif params['variation'] == 'varAnd':
                            next_gen = algorithms.varAnd(population = pop['out_nodes'][oi],
                                                         toolbox = varTbox['real'],
                                                         cxpb = params['prob_crossOver'],
                                                         mutpb = params['prob_mutation']
                            );
                        #generate random nodes
                        if params['randomNodesInject']:
                            print ">>inject";
                            rand_nodes = gen_random_nodes(layer = 'output',n = 10);
                            #add to next gen
                            next_gen.extend(rand_nodes);

                        #generate temporal nodes
                        if params['temporalNodesInject']:
                            print ">>Inject - Temporal patterning";
                            temp_nodes = gen_temporal_nodes(layer = 'output', n = 10);
                            #add to next gen
                            next_gen.extend(temp_nodes);

                        #Evaluate
                        evaluate_outputNodes(next_gen,oi);

                        #next generation
                        if next_gen == None:
                            next_gen = pop['out_nodes'][oi];

                        #selection of next population
                        pop['out_nodes'][oi] = tools.selTournament(next_gen,
                                                                   params['node_pop_size'],
                                                                   params['tourn_size']);

                    #selection of next representative
                    next_repr['out_nodes'][oi] = tbox.clone(tools.selBest(pop['out_nodes'][oi],1)[0]);


            ################## MODEL  ######################
            if s_key == 'model_pop':
                #evaluate the models
                specie = futures.map(evaluate_model_map, represntatives, specie);

                #should we optimise?
                next_gen = None;
                if not params['optimFreeze_model']:
                    print "-optim model";
                    if params['variation'] == 'varAnd':
                        #Variations (Cross over and Mutation)
                        next_gen = algorithms.varAnd(pop['model_pop'],
                                                     varTbox['binary'],
                                                     params['prob_crossOver'],
                                                     params['prob_mutation']
                        );
                    elif params['variation'] == 'varOr':
                        next_gen= algorithms.varOr(population = pop['model_pop'],
                                                   toolbox =    varTbox['binary'],
                                                   lambda_ = params['model_pop_size'],
                                                   cxpb =  1- params['prob_mutation'],
                                                   mutpb=   params['prob_mutation']
                        );

                    #evaluate
                    evaluate_model(next_gen);
                    #selection of next population
                    if next_gen == None:
                        next_gen = pop['model_pop'];
                    #select from next generation
                    pop['model_pop'] = tools.selTournament(next_gen,
                                                           params['model_pop_size'],
                                                           params['tourn_size']);

                #selection of next population
                next_repr['model'] = tbox.clone(tools.selBest(pop['model_pop'],1)[0]);

            ##################CONN_ACTIVE INPUT TO HIDDEN LAYER ######################
            if s_key == 'connActive_IH_pop':
                specie = futures.map(evaluate_connIH_map, represntatives, specie);

                #next generation
                next_gen = None;
                if not params['optimFreeze_connIH']:
                    print "-optim connIH";
                    #Variations (Cross over and Mutation)
                    if params['variation'] =='varOr': #select (parents + children)
                        next_gen = algorithms.varOr(population =pop['connActive_IH_pop'],
                                                    toolbox = varTbox['binary'],
                                                    lambda_ = params['conn_pop_size'],
                                                    cxpb = 1- params['prob_mutation'],
                                                    mutpb = params['prob_mutation']
                        );
                    elif params['variation'] == 'varAnd':
                        next_gen = algorithms.varAnd(population = pop['connActive_IH_pop'],
                                                     toolbox = varTbox['binary'],
                                                     cxpb  = params['prob_crossOver'],
                                                     mutpb = params['prob_mutation']);


                    #evaluate
                    #TODO

                    #next generation
                    if next_gen == None:
                        next_gen = pop['connActive_IH_pop'];
                    #selection of next population
                    pop['connActive_IH_pop'] = tools.selTournament(next_gen,
                                                                   params['conn_pop_size'],
                                                                   params['tourn_size']);
                #selection of next population
                next_repr['connActive_IH'] = tbox.clone(tools.selBest(pop['connActive_IH_pop'],1)[0]);


            ##################CONN_ACTIVE HIDDEN TO HIDDEN LAYER ######################
            if s_key == 'connActive_HH_pop' \
                    and params['numH'] > 1:

                specie = futures.map(evaluate_connHH_map, represntatives, specie);

                #should freeze optimisation?
                next_gen = None;
                if not params['optimFreeze_connHH']:
                    print "-optim connHH";
                    #Variations (Cross over and Mutation)
                    if params['variation'] =='varOr': #select (parents + children)

                        next_gen = algorithms.varOr(population =pop['connActive_HH_pop'],
                                                    toolbox = varTbox['binary'],
                                                    lambda_ = params['conn_pop_size'],
                                                    cxpb = 1-params['prob_mutation'],
                                                    mutpb = params['prob_mutation']
                        );
                    elif params['variation'] == 'varAnd':
                        next_gen = algorithms.varAnd(population = pop['connActive_HH_pop'],
                                                     toolbox = varTbox['binary'],
                                                     cxpb  = params['prob_crossOver'],
                                                     mutpb = params['prob_mutation']);

                    #evaluate
                    evaluate_connHH(next_gen);

                    #next generation
                    if next_gen == None:
                        next_gen = pop['connWeights_HH_pop'];
                    #selection of next population
                    pop['connActive_HH_pop'] = tools.selTournament(next_gen,
                                                                   params['conn_pop_size'],
                                                                   params['tourn_size']);

                #selection of next population
                next_repr['connActive_HH'] = tbox.clone(tools.selBest(pop['connActive_HH_pop'],1)[0]);

            ##################CONN_ACTIVE HIDDEN TO OUTPUT LAYER ######################
            if s_key == 'connActive_HO_pop':
                #evaluate and score
                specie = futures.map(evaluate_connHO_map, represntatives, specie)

                #should freeze optimisation?
                next_gen = None;
                if not params['optimFreeze_connHO']:
                    print "-optim connHO";
                    #Variations (Cross over and Mutation)
                    if params['variation'] =='varOr': #select (parents + children)
                        next_gen = algorithms.varOr(population =pop['connActive_HO_pop'],
                                                    toolbox = varTbox['binary'],
                                                    lambda_ = params['conn_pop_size'],
                                                    cxpb = 1-params['prob_mutation'],
                                                    mutpb = params['prob_mutation']
                        );
                    elif params['variation'] == 'varAnd':
                        next_gen= algorithms.varAnd(population = pop['connActive_HO_pop'],
                                                    toolbox = varTbox['binary'],
                                                    cxpb  = params['prob_crossOver'],
                                                    mutpb = params['prob_mutation']);


                    #evaluate
                    #TODO

                    if next_gen == None:
                        next_gen = pop['connActive_HO_pop'];

                    #selection of next population
                    pop['connActive_HO_pop'] = tools.selTournament(next_gen,
                                                                   params['conn_pop_size'],
                                                                   params['tourn_size']);

                #selection of next population
                next_repr['connActive_HO'] = tbox.clone(tools.selBest(pop['connActive_HO_pop'],1)[0]);

            ##################CONN_WEIGHTS INPUT TO HIDDEN LAYER ######################
            if s_key == 'connWeights_IH_pop':

                #evaluate and map
                specie = futures.map(evaluate_weightIH_map, represntatives, specie);


                #should freeze optimisation?
                next_gen = None;
                if not params['optimFreeze_weightsIH']:
                    print "-optim weightsIH";
                    #Variations (Cross over and Mutation)
                    if params['variation'] =='varOr': #select (parents + children)
                        next_gen = algorithms.varOr(population =pop['connWeights_IH_pop'],
                                                    toolbox = varTbox['real'],
                                                    lambda_ = params['conn_pop_size'],
                                                    cxpb = 1-params['prob_mutation'],
                                                    mutpb = params['prob_mutation']
                        );
                    elif params['variation'] == 'varAnd':
                        next_gen = algorithms.varAnd(population = pop['connWeights_IH_pop'],
                                                     toolbox = varTbox['real'],
                                                     cxpb  = params['prob_crossOver'],
                                                     mutpb = params['prob_mutation']);

                    #evaluate
                    evaluate_weightIH(next_gen);

                    #next generation
                    if next_gen == None:
                        next_gen = pop['connWeights_IH_pop'];

                    #selection of next population
                    pop['connWeights_IH_pop'] = tools.selTournament(next_gen,
                                                                    params['conn_pop_size'],
                                                                    params['tourn_size']);

                #selection of next population
                next_repr['connWeights_IH'] = tbox.clone(tools.selBest(pop['connWeights_IH_pop'],1)[0]);


            ##################CONN_WEIGHTS HIDDEN TO HIDDEN LAYER ######################
            if s_key == 'connWeights_HH_pop' \
                    and params['numH']>1:

                #evaluate and map
                specie = futures.map(evaluate_weightHH_map, represntatives, specie);

                #should freeze optimisation?
                next_gen = None;
                if not params['optimFreeze_weightsHH']:
                    print "-optim weightsHH";

                    #Variations (Cross over and Mutation)
                    if params['variation'] =='varOr': #select (parents + children)
                        next_gen = algorithms.varOr(population =pop['connWeights_HH_pop'],
                                                    toolbox = varTbox['real'],
                                                    lambda_ = params['conn_pop_size'],
                                                    cxpb = 1-params['prob_mutation'],
                                                    mutpb = params['prob_mutation']
                        );
                    elif params['variation'] == 'varAnd':
                        next_gen = algorithms.varAnd(population = pop['connWeights_HH_pop'],
                                                     toolbox = varTbox['real'],
                                                     cxpb  = params['prob_crossOver'],
                                                     mutpb = params['prob_mutation']);
                    #evaluate
                    evaluate_weightHH(next_gen);

                    #next generation
                    if next_gen == None:
                        next_gen = pop['connWeights_HH_pop'];

                    #selection of next population
                    pop['connWeights_HH_pop'] = tools.selTournament(next_gen,
                                                                    params['conn_pop_size'],
                                                                    params['tourn_size']);


                #selection of next population
                next_repr['connWeights_HH'] = tbox.clone(tools.selBest(pop['connWeights_HH_pop'],1)[0]);

            ##################CONN_WEIGHTS HIDDEN TO OUTPUT LAYER ######################
            if s_key == 'connWeights_HO_pop':
                #evaluate members of population
                specie = futures.map(evaluate_weightHO_map, represntatives, specie);

                #Should we optimise?
                next_gen = None;
                if not params['optimFreeze_weightsHO']:
                    print "-optim weightsHO";

                    #Variations (Cross over and Mutation)
                    if params['variation'] =='varOr': #select (parents + children)
                        try:
                            next_gen = algorithms.varOr(population =pop['connWeights_HO_pop'],
                                                        toolbox = varTbox['real'],
                                                        lambda_ = params['conn_pop_size'],
                                                        cxpb = 1-params['prob_mutation'],
                                                        mutpb = params['prob_mutation']
                            );
                        except ValueError:
                            ## Mutation Only ##
                            #get candidates and clone
                            candidates = [tbox.clone(w_ho) for w_ho in pop['connWeights_HO_pop']];
                            #mutate each one of them according to prob of params set.
                            mutants = [varTbox['real'].mutate(c_i) for c_i in candidates];
                            #items are returned as tuple from mutate operation so we unpack them
                            next_gen = [m_i[0] for m_i in mutants];

                    elif params['variation'] == 'varAnd':
                        try:
                            next_gen = algorithms.varAnd(population = pop['connWeights_HO_pop'],
                                                         toolbox = varTbox['real'],
                                                         cxpb  = params['prob_crossOver'],
                                                         mutpb = params['prob_mutation']);
                        except ValueError:
                            ## Mutation Only ##
                            #get candidates and clone
                            candidates = [tbox.clone(w_ho) for w_ho in pop['connWeights_HO_pop']];
                            #mutate each one of them according to prob of params set.
                            mutants = [varTbox['real'].mutate(c_i) for c_i in candidates];
                            #items are returned as tuple from mutate operation so we unpack them
                            next_gen = [m_i[0] for m_i in mutants];

                    #evaluate
                    evaluate_weightHO(next_gen);

                    #next generation
                    if next_gen == None:
                        next_gen = pop['connWeights_HO_pop'];

                    #selection of next population
                    pop['connWeights_HO_pop'] = tools.selTournament(next_gen,
                                                                    params['conn_pop_size'],
                                                                    params['tourn_size']);

                #selection of next population
                next_repr['connWeights_HO'] = tbox.clone(tools.selBest(pop['connWeights_HO_pop'],1)[0]);


        # Copy representation
        represntatives = tbox.clone(next_repr);
        #store best cost
        best_cost_lst.append(best_cost);


        #todo: take snapshot of generations (statistics)
        #todo: visualisation of current best network (with snapshots) - On Demand


    ################## *** OPTIMISATION COMPLETE *** ######################
    print "Optimisation Complete!!!";
    print "id(sol)", id(best_sol);
    print "sol:", best_sol;
    print "best_cost(train):", best_cost;
    #Test
    train_err = best_model.evaluate(train_set);
    best_model.flush();
    test_err = best_model.evaluate(test_set);
    best_cost_test = test_err;
    print "best_cost(train):",train_err;
    print "best_cost(test):",test_err;
    best_model.flush();

    ##STORE STATISTICS ###

    # - store records
    # - get time stamp
    timestamp = time.ctime();
    timestamp = timestamp.replace(' ','_');
    timestamp = timestamp.replace(':','');


    #Form pre string for filenames
    preString = dataset_name + '_' +  timestamp+ '_';
    # - Store statistics
    DataFrame( data = best_cost_lst, columns = ['best_cost']).to_csv(preString+'gen_fitness.csv');
    for k, v in snapshot.iteritems():
        DataFrame(data = snapshot[k], columns = [k]).to_csv(preString+ k + '.csv');

    #-store best models performance
    performance = {'train_err': [train_err],
                   'test_err': [test_err]};
    DataFrame(performance).to_csv(preString+'Performance.csv');

    # store settings
    exp_params = dict();
    exp_params['NGEN'] = params['NGEN'];
    exp_params['numI'] = params['numI'];
    exp_params['numH'] = params['numH'];
    exp_params['numO'] = params['numO'];
    exp_params['optimFreeze_weightsHH'] = params['optimFreeze_weightsHH'];
    exp_params['optimFreeze_weightsHO'] = params['optimFreeze_weightsHO'];
    exp_params['optimFreeze_weightsIH'] = params['optimFreeze_weightsIH'];
    exp_params['optimFreeze_connIH'] = params['optimFreeze_connIH'];
    exp_params['optimFreeze_connHH'] = params['optimFreeze_connHH'];
    exp_params['optimFreeze_connHO'] = params['optimFreeze_connHO'];
    exp_params['model_pop_size'] = params['model_pop_size'];
    exp_params['node_pop_size'] = params['node_pop_size'];
    exp_params['conn_pop_size'] = params['conn_pop_size'];
    exp_params['initConnectivity'] = params['initConnectivity'];
    exp_params['initTransferFn'] = params['initTransferFn'];
    exp_params['prodConstantMax'] = params['prodConstantMax'];
    #SAVE EXP Params
    expParamsFile = open(preString+'EXP_params.dat','w');
    pickle.dump(exp_params,expParamsFile );
    expParamsFile.close();

    ####STORE BEST MODEL###
    fstore = open(dataset_name+'best_model'+timestamp+'.dat', 'w');
    pickle.dump(
        {'best_sol': best_sol,
         'best_cost': best_cost},fstore);

    fstore.close();

    ################## RETURN ######################

    #return best_model;



