__author__ = 'Abdullah'

import sys;
sys.path.insert(0, "../optim");
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../visualisation");


from visualiseNDMNet import *;

from deap import tools,creator, base;
import numpy as np;
import pyNDMOptim;
import ndmModel;
import random;

import activation_fn;
import output_fn;
import datasets;
from PyQt4 import QtCore, QtGui;

#CONSTANTS
NODEPARAMS = 5;
MODELPARAMS = 3;

#PARAMETERS
params = dict();
params['numI'] = 2;
params['numH'] = 5;
params['numO'] = 1;

params['activation_fns'] = [activation_fn.euclidean_dist, activation_fn.inner_prod, activation_fn.min];
params['output_fns']=  [output_fn.identity_fn, output_fn.sigmoid_fn, output_fn.gaussian_fn];
params['initConnectivity'] = 'full'; #random or full
params['initTransferFn'] = 'random'; #all or random

#INTIALISATION
#create toolbox
tbox = base.Toolbox();

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
               prodConstant = 5,
               id = None);

#Register to the toolbox
# - Attributes
tbox.register("float_attr", random.random);
tbox.register("bool_attr", random.randint, 0, 1);

################################ 1- Node ################################################################
tbox.register("node", tools.initRepeat, creator.node, tbox.float_attr, NODEPARAMS);

################################ Connections/Topology #################################################
# - 2- Connectivity (Input to hidden units)
if params['initConnectivity'] == 'random':
    tbox.register("connActive_IH", pyNDMOptim.initRandomConnectivity, creator.ConnActive,
                       params['numI'],
                       params['numH']);
    tbox.register("connActive_HO", pyNDMOptim.initRandomConnectivity, creator.ConnActive,
                       params['numH'],
                       params['numO']);
    tbox.register("connActive_HH", pyNDMOptim.initRandomConnectivity, creator.ConnActive,
                       params['numH'],
                       params['numH']);

elif params['initConnectivity'] == 'full':
    tbox.register("connActive_IH", pyNDMOptim.initFullConnectivity, creator.ConnActive,
                       params['numI'],
                       params['numH']);
    tbox.register("connActive_HO", pyNDMOptim.initFullConnectivity, creator.ConnActive,
                       params['numH'],
                       params['numO']);
        # - Connectivity (Hidden to hidden layer)
    tbox.register("connActive_HH", pyNDMOptim.initFullConnectivity, creator.ConnActive,
                       params['numH'],
                       params['numH']);

# - 3- Connection Weights (Input to hidden layer, hidden layer to output layer)
tbox.register("connWeights_IH", pyNDMOptim.initRandomWeights, creator.ConnWeights,
                       params['numI'],
                       params['numH']);
tbox.register("connWeights_HO", pyNDMOptim.initRandomWeights, creator.ConnWeights,
                       params['numH'],
                       params['numO']);



    # - Connection weights (hidden to hidden layer)
tbox.register("connWeights_HH", pyNDMOptim.initRandomWeights, creator.ConnWeights,
                       params['numH'],
                       params['numH']);

# - Model
tbox.register("model", tools.initRepeat, creator.model, tbox.bool_attr,MODELPARAMS)



# CREATE NODES
hidden_nodes = [tbox.node() for _ in xrange(params['numH'])];
out_nodes = [tbox.node() for _ in xrange(params['numO'])];

# CREATE CONN ACTIVE
connActiveIH = tbox.connActive_IH();
connActiveHH = tbox.connActive_HH();
connActiveHO = tbox.connActive_HO();

weightsIH = tbox.connWeights_IH();
weightsHH = tbox.connWeights_HH();
weightsHO = tbox.connWeights_HO();

model = tbox.model();

print "model",model;
print "connActiveIH", connActiveIH;
print "connActiveHH", connActiveHH;
print "connActiveHO", connActiveHO;

print "connWeightsIH", weightsIH;
print "connWeightsHH", weightsHH;
print "connWeightsHO", weightsHO;

# - Transfer functions
if params['initTransferFn'] == 'all': #i.e. all possible combinations (no duplicates)

        print ">>All Transfer functions";

        tfs_comb = [(w,n) for w in params['activation_fns'] for n in params['output_fns']];

        print ">>hidden nodes";
        for i, (node,tf) in enumerate(zip(hidden_nodes,tfs_comb)):
                #output functions
                node.activation_fn = tf[0];
                node.output_fn = tf[1];

                print "node",i;
                print "output_fn",node.output_fn;
                print "activation_fn", node.activation_fn;

        print ">>output nodes";
        for i, (node,tf) in enumerate(zip(out_nodes,tfs_comb)):
                #output functions
                node.activation_fn = tf[0];
                node.output_fn = tf[1];

                print "node",i;
                print "output_fn",node.output_fn;
                print "activation_fn", node.activation_fn;


elif params['initTransferFn'] == 'random': #i.e. random, with possibility of duplicates

        print ">>Random Transfer functions";
        print ">>hidden nodes";
        for i, node in enumerate(hidden_nodes):
            #output function
            outfn = random.randint(0,len(params['output_fns'])-1);
            infn = random.randint(0,len(params['activation_fns'])-1);

            ## Reference the function directly
            node.output_fn = params['output_fns'][outfn];
            node.activation_fn = params['activation_fns'][infn];

            print "Node",i;
            print "output_fn",node.output_fn;
            print "activation_fn", node.activation_fn;

        print ">>output nodes";
        for i, node in enumerate(out_nodes):
            #output function
            outfn = random.randint(0,len(params['output_fns'])-1);
            infn = random.randint(0,len(params['activation_fns'])-1);

            ## Reference the function directly
            node.output_fn = params['output_fns'][outfn];
            node.activation_fn = params['activation_fns'][infn];

            print "Node",i;
            print "output_fn",node.output_fn;
            print "activation_fn", node.activation_fn;




#TODOLIST
#TODO: (Tommorow) Test build model from signature


#select componentsesentatives
components = dict();
components['hidden_nodes'] = hidden_nodes;
components['out_nodes'] = out_nodes;
#select random connections and weights
components['model'] = model;
components['connActive_IH'] = connActiveIH;
components['connActive_HH'] = connActiveHH;
components['connActive_HO'] = connActiveHO;
components['connWeights_IH'] = weightsIH;
components['connWeights_HH'] = weightsHH;
components['connWeights_HO'] = weightsHO;




ndmmodel = ndmModel.ndmModel(params['numI'], params['numH'], params['numO'], components);
print "inputs:", datasets.XOR['IN'][0];
o = ndmmodel.stimulate(datasets.XOR['IN'][0]);
o2 = ndmmodel.stimulate(datasets.XOR['IN'][1]);

print "net out:", o;
print "out:", datasets.XOR['OUT'][0];
print "err:", ndmmodel.evaluate(datasets.XOR);

app = QtGui.QApplication(sys.argv);
QtCore.qsrand(QtCore.QTime(0,0,0).secsTo(QtCore.QTime.currentTime()));
widget2 = GraphWidget(ndmmodel,'Train Set');
widget2.show();
sys.exit(app.exec_());
