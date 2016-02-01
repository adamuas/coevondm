__author__ = 'Abdullah'


from deap import base, creator, tools;
import numpy as np;
import random;

#Fitness class
creator.create("fitness", base.Fitness, weights=(-1, -1));

#Node class
creator.create("node", np.ndarray,
               fitness=creator.fitness,
               id=None ,
               node_type=None,
               activation_fn=None,
               output_fn=None);

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




def init_toolbox(params, NODEPARAMS=5, MODELPARAMS =3, toolbox = None):
    """
    Initialise DEAP toolbox
    """

    if toolbox == None:
        #Register to the toolbox
        toolbox = base.Toolbox();
    else:
        pass;

    # - Attributes
    toolbox.register("float_attr", np.random.rand);
    toolbox.register("bool_attr", random.randint, 0, 1);
    toolbox.register("ones", random.randint, 1,1);
    toolbox.register("zeros", random.randint, 0,0);
    ################################ 1- Node ################################################################
    toolbox.register("node", tools.initRepeat, creator.node, toolbox.float_attr, NODEPARAMS);

    ################################ Connections/Topology #################################################
    # - 2- Connectivity (Input to hidden units)
    if params['initConnectivity'] == 'full':
        #IH
        toolbox.register("connActive_IH", tools.initRepeat, creator.ConnActive, toolbox.ones,
                           params['numI'] * params['numH']);
        #HO
        toolbox.register("connActive_HO", tools.initRepeat,creator.ConnActive, toolbox.ones,
                           params['numH'] * params['numO']);
        #HH
        toolbox.register("connActive_HH", tools.initRepeat, creator.ConnActive,toolbox.ones,
                           params['numH'] * params['numH']);

    elif params['initConnectivity'] == 'random':
        #IH
        toolbox.register("connActive_IH", tools.initRepeat, creator.ConnActive, toolbox.bool_attr,
                           params['numI'] * params['numH']);
        #HO
        toolbox.register("connActive_HO", tools.initRepeat, creator.ConnActive,toolbox.bool_attr,
                           params['numH'] * params['numO']);
        #HH
        toolbox.register("connActive_HH", tools.initRepeat, creator.ConnActive,toolbox.bool_attr,
                           params['numH'] * params['numH']);

    # elif params['initConnectivity'] == 'none':
    #     #IH
    #     toolbox.register("connActive_IH", tools.initZeroConnectivity, creator.ConnActive,
    #                        params['numI'] , params['numH']);
    #     #HO
    #     toolbox.register("connActive_HO", tools.initZeroConnectivity, creator.ConnActive,
    #                        params['numH'] , params['numO']);
    #     #HH
    #     toolbox.register("connActive_HH", tools.initZeroConnectivity, creator.ConnActive,
    #                        params['numH'], params['numH']);


    # - Connection weights (input to hidden layer)
    toolbox.register("connWeights_IH", tools.initRepeat, creator.ConnWeights, toolbox.float_attr,
                           params['numI'] * params['numH']);
    # - Connection weights (hidden to output layer)
    toolbox.register("connWeights_HO", tools.initRepeat, creator.ConnWeights,toolbox.float_attr,
                           params['numH'] * params['numO']);
    # - Connection weights (hidden to hidden layer)
    toolbox.register("connWeights_HH", tools.initRepeat, creator.ConnWeights, toolbox.float_attr,
                           params['numH'] * params['numH']);
    #- Model
    toolbox.register("model", tools.initRepeat, creator.model, toolbox.bool_attr,MODELPARAMS)


    #return the toolbox
    return toolbox;

def init_tf_fn_random(nodes,params,layer):
        """
        Initialise the nodes with the set of transfer functions available
        """

        for k,node in nodes.iteritems():

            for n in node:

                if layer == 'hidden':
                    #output function
                    outfn = random.randint(0,len(params['hl_output_fns'])-1);
                    #activation function
                    infn = random.randint(0,len(params['hl_activation_fns'])-1);

                    ## Reference the function directly
                    n.output_fn = params['hl_output_fns'][outfn];
                    n.activation_fn = params['hl_activation_fns'][infn];
                elif layer == 'output':
                    #output function
                    outfn = random.randint(0,len(params['ol_output_fns'])-1);
                    #activation function
                    infn = random.randint(0,len(params['ol_activation_fns'])-1);

                    ## Reference the function directly
                    n.output_fn = params['ol_output_fns'][outfn];
                    n.activation_fn = params['ol_activation_fns'][infn];


def init_tf_fn_forPop(nodes,params,layer):
    """

    Initialise
    """

    for n in nodes:

            if layer == 'hidden':
                    #output function
                    outfn = random.randint(0,len(params['hl_output_fns'])-1);
                    #activation function
                    infn = random.randint(0,len(params['hl_activation_fns'])-1);

                    ## Reference the function directly
                    n.output_fn = params['hl_output_fns'][outfn];
                    n.activation_fn = params['hl_activation_fns'][infn];
            elif layer == 'output':
                    #output function
                    outfn = random.randint(0,len(params['ol_output_fns'])-1);
                    #activation function
                    infn = random.randint(0,len(params['ol_activation_fns'])-1);

                    ## Reference the function directly
                    n.output_fn = params['ol_output_fns'][outfn];
                    n.activation_fn = params['ol_activation_fns'][infn];

#
# def init_tf_fn_temporal(nodes,tf,layer):
#     """
#
#     Initialise for temporal patterning
#     """
#
#     for n in nodes:
#
#             if layer == 'hidden':
#                     #output function
#                     outfn = random.randint(0,len(tf['hl_output_fns'])-1);
#                     #activation function
#                     infn = random.randint(0,len(tf['hl_activation_fns'])-1);
#
#                     ## Reference the function directly
#                     n.output_fn = tf['hl_output_fns'][outfn];
#                     n.activation_fn = tf['hl_activation_fns'][infn];
#             elif layer == 'output':
#                     #output function
#                     outfn = random.randint(0,len(tf['ol_output_fns'])-1);
#                     #activation function
#                     infn = random.randint(0,len(tf['ol_activation_fns'])-1);
#
#                     ## Reference the function directly
#                     n.output_fn = tf['ol_output_fns'][outfn];
#                     n.activation_fn = tf['ol_activation_fns'][infn];




def init_all_transferFns(nodes, params, layer):
        """
        Initialise the transfer functions
        """
        if layer == 'hidden':
            tfs_comb = [(w,n) for w in params['hl_activation_fns'] for n in params['hl_output_fns']];

        elif layer == 'output':
            tfs_comb = [(w,n) for w in params['ol_activation_fns'] for n in params['ol_output_fns']];


        for k,node in nodes.iteritems():

            for i, n in enumerate(node):
                tf = tfs_comb[i];
                #output function
                outfn =tf[1];
                infn = tf[0];

                ## Reference the function directly
                n.output_fn =outfn;
                n.activation_fn = infn;


