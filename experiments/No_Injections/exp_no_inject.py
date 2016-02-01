__author__ = 'Abdullah'



import sys;
sys.path.insert(0, "../../optim");
sys.path.insert(0, "../../datasets");
sys.path.insert(0, "../../core");
sys.path.insert(0, "../../visualisation");
sys.path.insert(0, "../../tools");

import ndmCoevoOptim;
from benchmark import proben1_bechmark as proben1;
from benchmark import lab_bencmark;
import kfold;
import profile;




def test_xor(RUNS = 5,N = 10):
    """
    test for XOR dataset
    """

    lab_data = lab_bencmark();

    #dataset divided into train and test as ratio of 70:30
    D = lab_data.xor(N);
    netConfig = {'numI': D['test']['INFO']['num_inputs'],
                 'numO': D['test']['INFO']['num_outputs'],
                 'numH': 2
    };
    coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name = 'XOR', train_set = D['train'],
                                            valid_set = D['test'],
                                            test_set = D['test']);
    for ri in xrange(RUNS):
        #disable random inject
        coevo.params['randomNodesInject'] = False;
        coevo.init_populations();
        # coevo.params['numI'] =
        m = coevo.coevolve();
        #do signature extraction on the model
        
        #store signatures to the respective folder

def test_iris(RUNS = 10):
    """
    test for the iris dataset
    """

    lab_data = lab_bencmark();

    print ">>>IRIS";

    D = lab_data.iris();
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': 4,
                 'numO': 1,
                 'numH': 2
    };
    for ri in xrange(RUNS):
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name = 'IRIS',
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);
        #disable random inject
        coevo.params['randomNodesInject'] = False;
        coevo.init_populations();
        coevo.coevolve();

    del lab_data;

def test_sonar(RUNS = 10):
    """
    SONAR
    """

    lab_data = lab_bencmark();

    D = lab_data.sonar();
    coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name = 'SONAR', train_set = D['train'], test_set = D['test']);

    for ri in xrange(RUNS):
        coevo.init_populations();
        #disable random inject
        coevo.params['randomNodesInject'] = False;
        m = coevo.coevolve();

    del lab_data;

def test_glass(RUNS = 10):
    """
    GLASS
    """

    proben = proben1();
    D = proben.glass();
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['test']['INFO']['num_inputs'],
                 'numO': D['test']['INFO']['num_outputs'],
                 'numH': 2
    };
    for ri in xrange(RUNS):
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name =  D['name'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);

        #disable random inject
        coevo.params['randomNodesInject'] = False;

        coevo.init_populations();
        m = coevo.coevolve();


    del proben;


def test_aussie_cc(RUNS = 5):
    """
    australian cc
    """

    proben = proben1();
    D = proben.australian_cc();
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['test']['INFO']['num_inputs'],
                 'numO': D['test']['INFO']['num_outputs'],
                 'numH': 2
    };

    for ri in xrange(RUNS):
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name =  D['name'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);

        #disable random inject
        coevo.params['randomNodesInject'] = False;

        coevo.init_populations();
        m = coevo.coevolve();

    del proben;

def test_cancer(RUNS = 10):
    """
    Breast Cancer - Wisconsin
    """

    proben = proben1();
    D = proben.breast_cancer();
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['test']['INFO']['num_inputs'],
                 'numO': D['test']['INFO']['num_outputs'],
                 'numH': 2
    };

    for ri in xrange(RUNS):
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name =  D['name'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);

        #disable random inject
        coevo.params['randomNodesInject'] = False;

        coevo.init_populations();
        m = coevo.coevolve();


    del proben;

def test_heart(RUNS = 10):
    """
    Heart
    """

    proben = proben1();
    D = proben.heart();
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['test']['INFO']['num_inputs'],
                 'numO': D['test']['INFO']['num_outputs'],
                 'numH': 2
    };
    for ri in xrange(RUNS):
       coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name =  D['name'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);

        #disable random inject
       coevo.params['randomNodesInject'] = False;

       coevo.init_populations();
       m = coevo.coevolve();

    del proben;





def test_diabetes(RUNS = 10):
    """
    Diabetes
    """


    proben = proben1();
    D = proben.diabetes();
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['test']['INFO']['num_inputs'],
                 'numO': D['test']['INFO']['num_outputs'],
                 'numH': 2
    };
    for ri in xrange(RUNS):
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name =  D['name'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);

        #disable random inject
        coevo.params['randomNodesInject'] = False;

        coevo.init_populations();
        m = coevo.coevolve();


    del proben;



def main():
    """
    Establish results for the control experiment
    """

    ## XOR ##
    #test_xor();

    ## IRIS ##
    test_iris();
    # ## SONAR ##
    # test_sonar();
    # ## AUSSIE CC##
    #test_aussie_cc();
    # ## CANCER ##
    #test_cancer();
    # ## HEART ##
    #test_heart();
    ## GLASS ###
    #test_glass();

    # ## DIABETEST ##
    #test_diabetes();




if __name__ == '__main__':
    main();

    # profile.run("main()");


