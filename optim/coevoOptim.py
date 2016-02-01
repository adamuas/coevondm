__author__ = 'Abdullah'

#Scalable Computing
scoop = True;
multiprocessing = True;


from deap import tools, base, creator;

try:
    from scoop import futures;
    #import dtm;
except ImportError:
   scoop = False;
   #dtm  = False


import sys;

sys.path.insert(0, "../");  #use default settings
sys.path.insert(0, "../core");
sys.path.insert(0, "../visualisation");
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../cuda");
sys.path.insert(0, "../tools");




def getToolbox(params):
    """
    Returns a dictionary of toolboxes, one for real parameters, and the other for binary operators
    """
    #Variation toolbox


    VarToolbox = {
        'real': base.Toolbox(),
        'binary' : base.Toolbox()
    };


    ### >>> Toolbox for real values ###
    #- MUTATE
    VarToolbox['real'].register("mutate", tools.mutGaussian,
                     mu = params['gaus_mut_mean'],
                     sigma = params['gaus_mut_std'],
                     indpb = params['prob_mut_indp']);
    #- MATE
    VarToolbox['real'].register("mate", tools.cxTwoPoint);
    #-SELECTION
    VarToolbox['real'].register("select", tools.selTournament, tournsize = params['tourn_size']);
    #-SCOOP (Scalable Concurent Operations in Python)
    if scoop:
        VarToolbox['real'].register("map", futures.map);
    # if dtm:
    #     VarToolbox['real'].register("map", dtm.map);
    ### >>> Toolbox for binary values ###
    #- MUTATE
    VarToolbox['binary'].register("mutate", tools.mutFlipBit,
                     indpb = params['prob_mut_indp']);
    #- MATE
    VarToolbox['binary'].register("mate", tools.cxTwoPoint);
    #-SELECTION
    VarToolbox['binary'].register("select", tools.selTournament, tournsize = params['tourn_size']);
    #-SCOOP (Scalable Concurent Operations in Python)
    if scoop:
        VarToolbox['binary'].register("map", futures.map);
    # if dtm:
    #     VarToolbox['binary'].register("map", dtm.map);



    return VarToolbox;

