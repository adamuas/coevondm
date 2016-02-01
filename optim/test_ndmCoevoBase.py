__author__ = 'Abdullah'
import sys;
sys.path.insert(0, "../");  #use default settings
sys.path.insert(0, "../core");
sys.path.insert(0, "../datasets");

from deap import tools, base, creator;
import ndmCoevoBase;
import commons;
import ndmModel;
import random;

import activation_fn;
import output_fn;
import datasets;

#PARAMETERS
params = dict();
params['numI'] = 2;
params['numH'] = 1;
params['numO'] = 1;

params['activation_fns'] = [activation_fn.euclidean_dist, activation_fn.inner_prod, activation_fn.min];
params['output_fns']=  [output_fn.identity_fn, output_fn.sigmoid_fn, output_fn.gaussian_fn];
params['initConnectivity'] = 'full'; #random or full
params['initTransferFn'] = 'random'; #all or random

tbox = ndmCoevoBase.init_toolbox(params);



### TEST MODEL Creation ###
# CREATE NODES
hidden_nodes = [tbox.node() for _ in xrange(params['numH'])];
out_nodes = [tbox.node() for _ in xrange(params['numO'])];



# CREATE CONN ACTIVE
connActiveIH = commons.matrixForm(tbox.connActive_IH(),params['numI'], params['numH']);
connActiveHH = commons.matrixForm(tbox.connActive_HH(), params['numH'], params['numH']);
connActiveHO = commons.matrixForm(tbox.connActive_HO(), params['numH'], params['numO']);

weightsIH = commons.matrixForm(tbox.connWeights_IH(), params['numI'], params['numH']);
weightsHH = commons.matrixForm(tbox.connWeights_HH(), params['numH'], params['numH']);
weightsHO = commons.matrixForm(tbox.connWeights_HO(), params['numH'], params['numO']);

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